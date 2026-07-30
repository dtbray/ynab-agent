"""Transaction polling service for unapproved transaction reminders."""

import asyncio
import logging
from typing import Optional

from ynab_agent.api.client import YnabClient
from ynab_agent.notifications.discord import DiscordNotifier, TransactionNotification
from ynab_agent.config import settings
from ynab_agent.services.polling import TransactionPollStore

logger = logging.getLogger(__name__)


class TransactionPoller:
    """Polls YNAB for unapproved transactions and sends reminders."""
    
    def __init__(
        self,
        db_manager: TransactionPollStore,
        discord_notifier: Optional[DiscordNotifier] = None
    ):
        self.db = db_manager
        self.notifier = discord_notifier or DiscordNotifier()
        self.running = False
    
    async def sync_and_check(self) -> list[TransactionNotification]:
        """Sync transactions and return unapproved ones needing notification."""
        logger.info("Starting YNAB sync...")
        
        async with YnabClient() as client:
            # Get unapproved transactions from API
            unapproved = await client.get_unapproved_transactions()
            logger.info(f"Found {len(unapproved)} unapproved transactions from YNAB")
            
            # Get budget ID from config
            budget_id = settings.ynab_plan_id
            if budget_id == "last-used":
                # The API returns transactions with their budget_id
                budget_id = unapproved[0].get("budget_id", "default") if unapproved else "default"
            
            # Save to database
            await self.db.save_transactions(budget_id, unapproved)
            
            # Get transactions that haven't been notified yet
            to_notify = await self.db.get_unnotified_transactions()
            logger.info(f"{len(to_notify)} transactions need notification")
            
            # Convert to notification format
            notifications = [
                TransactionNotification(
                    id=txn["id"],
                    date=txn["date"],
                    amount=txn["amount"],
                    memo=txn["memo"],
                    account_name=txn.get("account_name", "Unknown"),
                    payee_name=txn.get("payee_name")
                )
                for txn in to_notify
            ]
            
            return notifications
    
    async def send_reminders(self, transactions: list[TransactionNotification]) -> int:
        """Send reminders and mark as notified."""
        if not transactions:
            return 0
        
        # Send Discord notification
        await self.notifier.send(transactions)
        
        # Mark all as notified
        for txn in transactions:
            await self.db.mark_notified(txn.id)
        
        logger.info(f"Sent reminders for {len(transactions)} transactions")
        return len(transactions)
    
    async def run_once(self) -> int:
        """Run a single poll cycle. Returns count of reminders sent."""
        try:
            notifications = await self.sync_and_check()
            return await self.send_reminders(notifications)
        except Exception as e:
            logger.error(f"Error during poll cycle: {e}", exc_info=True)
            return 0
    
    async def run_continuous(self, interval_minutes: Optional[int] = None):
        """Run continuous polling loop."""
        interval = interval_minutes or settings.poll_interval_minutes
        self.running = True
        
        logger.info(f"Starting continuous polling every {interval} minutes")
        
        while self.running:
            await self.run_once()
            
            # Sleep in 1-second chunks to allow graceful shutdown
            for _ in range(interval * 60):
                if not self.running:
                    break
                await asyncio.sleep(1)
        
        logger.info("Polling stopped")
    
    def stop(self):
        """Signal the poller to stop."""
        self.running = False
