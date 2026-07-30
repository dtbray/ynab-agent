"""Discord notification service for transaction reminders."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from ynab_agent.config import settings


@dataclass
class TransactionNotification:
    """Transaction to be notified about."""
    id: str
    date: str
    amount: int  # milliunits
    memo: Optional[str]
    account_name: str
    payee_name: Optional[str] = None
    
    @property
    def amount_dollars(self) -> Decimal:
        """Convert milliunits to dollars."""
        return Decimal(self.amount) / 1000


class DiscordNotifier:
    """Sends Discord notifications for unapproved transactions."""
    
    def __init__(self, target: Optional[str] = None):
        self.target = target or settings.discord_target
        self.enabled = settings.discord_enabled
    
    def format_message(self, transactions: list[TransactionNotification]) -> str:
        """Format transactions into Discord message."""
        if not transactions:
            return ""
        
        lines = [
            "🔔 **YNAB: Transactions Needing Approval**",
            "",
            f"Found {len(transactions)} transaction(s) to review:",
            ""
        ]
        
        for txn in transactions[:10]:  # Limit to 10 per message
            amount_str = f"${abs(txn.amount_dollars):,.2f}"
            prefix = "➖" if txn.amount < 0 else "➕"
            
            desc = txn.memo or txn.payee_name or "Unknown"
            lines.append(f"{prefix} **{amount_str}** — {desc}")
            lines.append(f"   📅 {txn.date} | {txn.account_name}")
            lines.append("")
        
        if len(transactions) > 10:
            lines.append(f"... and {len(transactions) - 10} more")
        
        lines.append("")
        lines.append("Approve these in YNAB to stop these reminders.")
        
        return "\n".join(lines)
    
    async def send(self, transactions: list[TransactionNotification]) -> bool:
        """Send notification to Discord."""
        if not self.enabled or not transactions:
            return False
        
        message = self.format_message(transactions)
        
        # Emit a structured line for an optional supervising process.
        print(f"DISCORD_NOTIFICATION:{self.target}:{message}")
        return True
