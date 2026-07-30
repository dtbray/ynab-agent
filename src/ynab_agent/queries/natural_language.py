"""Natural language interface for budget queries using LLM."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from ynab_agent.db.manager import DatabaseManager


@dataclass
class BudgetQueryResult:
    """Result of a natural language query."""
    query: str
    sql: str
    summary: str
    data: list[dict]
    total_amount: Optional[Decimal] = None


class NaturalLanguageQuerier:
    """Processes natural language queries against budget data."""
    
    COMMON_PATTERNS = {
        "unapproved": "SELECT COUNT(*) as count FROM transactions WHERE approved IS FALSE AND deleted IS FALSE",
        "pending": "SELECT COUNT(*) as count FROM transactions WHERE cleared = 'uncleared' AND deleted IS FALSE",
        "this month": "WHERE date >= date('now', 'start of month')",
        "last month": "WHERE date >= date('now', '-1 month', 'start of month') AND date < date('now', 'start of month')",
        "top spending": "ORDER BY ABS(amount) DESC LIMIT 10",
    }
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    async def query(self, question: str) -> BudgetQueryResult:
        """Process a natural language question and return results."""
        
        question_lower = question.lower().strip()
        
        # Pattern matching for common queries
        if "unapproved" in question_lower or "needs approval" in question_lower:
            return await self._get_unapproved_transactions()
        
        elif "pending" in question_lower:
            return await self._get_pending_transactions()
        
        elif "spent" in question_lower and "this month" in question_lower:
            return await self._get_spending_this_month()
        
        elif "balance" in question_lower:
            return await self._get_account_balances()
        
        elif "top" in question_lower and ("spending" in question_lower or "expenses" in question_lower):
            return await self._get_top_spending()
        
        else:
            # Generic fallback
            return BudgetQueryResult(
                query=question,
                sql="-- No pattern matched",
                summary=f"I don't have a pattern for: '{question}'. Try: unapproved, pending, spending this month, account balances, or top spending.",
                data=[]
            )
    
    async def _get_unapproved_transactions(self) -> BudgetQueryResult:
        """Get unapproved transactions."""
        sql = """
            SELECT t.id, t.date, t.amount, t.memo, a.name as account_name
            FROM transactions t
            JOIN accounts a ON t.account_id = a.id
            WHERE t.approved IS FALSE AND t.deleted IS FALSE
            ORDER BY t.date DESC
        """
        
        data = await self.db.fetch_all(sql)
        
        total = sum(Decimal(r["amount"]) for r in data) if data else Decimal(0)
        count = len(data)
        
        return BudgetQueryResult(
            query="Show unapproved transactions",
            sql=sql,
            summary=f"Found {count} unapproved transactions totaling ${total/1000:.2f}",
            data=data,
            total_amount=total / 1000
        )
    
    async def _get_spending_this_month(self) -> BudgetQueryResult:
        """Get spending for current month."""
        sql = """
            SELECT 
                c.name as category,
                SUM(ABS(t.amount)) as total,
                COUNT(*) as count
            FROM transactions t
            JOIN categories c ON t.category_id = c.id
            WHERE t.date >= date('now', 'start of month')
                AND t.amount < 0
                AND t.deleted IS FALSE
                AND t.approved IS TRUE
            GROUP BY c.name
            ORDER BY total DESC
            LIMIT 20
        """
        
        data = await self.db.fetch_all(sql)
        
        total = sum(Decimal(r["total"]) for r in data) if data else Decimal(0)
        
        return BudgetQueryResult(
            query="Spending this month",
            sql=sql,
            summary=f"This month's spending: ${total/1000:.2f} across {len(data)} categories",
            data=data,
            total_amount=total / 1000
        )
    
    async def _get_account_balances(self) -> BudgetQueryResult:
        """Get account balances."""
        sql = """
            SELECT name, type, balance / 1000.0 as balance_dollars
            FROM accounts
            WHERE closed IS FALSE AND deleted IS FALSE
            ORDER BY balance DESC
        """
        
        data = await self.db.fetch_all(sql)
        
        total = sum(Decimal(r["balance_dollars"]) for r in data) if data else Decimal(0)
        
        return BudgetQueryResult(
            query="Account balances",
            sql=sql,
            summary=f"Total balance across {len(data)} accounts: ${total:.2f}",
            data=data,
            total_amount=total
        )
    
    async def _get_top_spending(self) -> BudgetQueryResult:
        """Get top spending transactions."""
        sql = """
            SELECT 
                t.date,
                t.memo,
                ABS(t.amount) / 1000.0 as amount,
                c.name as category,
                a.name as account
            FROM transactions t
            JOIN categories c ON t.category_id = c.id
            JOIN accounts a ON t.account_id = a.id
            WHERE t.amount < 0 AND t.deleted IS FALSE
            ORDER BY ABS(t.amount) DESC
            LIMIT 10
        """
        
        data = await self.db.fetch_all(sql)
        
        return BudgetQueryResult(
            query="Top spending",
            sql=sql,
            summary=f"Top {len(data)} largest transactions",
            data=data
        )
    
    async def _get_pending_transactions(self) -> BudgetQueryResult:
        """Get pending (uncleared) transactions."""
        sql = """
            SELECT t.id, t.date, t.amount, t.memo, a.name as account_name
            FROM transactions t
            JOIN accounts a ON t.account_id = a.id
            WHERE t.cleared = 'uncleared' AND t.deleted IS FALSE
            ORDER BY t.date DESC
        """
        
        data = await self.db.fetch_all(sql)
        
        count = len(data)
        
        return BudgetQueryResult(
            query="Pending transactions",
            sql=sql,
            summary=f"Found {count} pending (uncleared) transactions",
            data=data
        )
