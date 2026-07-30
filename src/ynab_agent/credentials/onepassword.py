"""1Password credential manager for secure token storage."""

import os
import subprocess
from typing import Optional


class OnePasswordCredentialManager:
    """Manages YNAB credentials via 1Password CLI."""
    
    def __init__(
        self,
        vault: Optional[str] = None,
        item: Optional[str] = None,
        field: str = "password"
    ):
        self.vault = vault or os.getenv("OP_VAULT_NAME", "")
        self.item = item or os.getenv("OP_ITEM_NAME", "")
        self.field = field
    
    def get_ynab_token(self) -> str:
        """Retrieve YNAB access token from 1Password."""
        if not self.vault or not self.item:
            raise RuntimeError(
                "OP_VAULT_NAME and OP_ITEM_NAME are required for 1Password auth"
            )
        cmd = [
            "op", "item", "get", self.item,
            "--vault", self.vault,
            "--field", self.field,
            "--reveal"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to retrieve token: {e.stderr}") from e
