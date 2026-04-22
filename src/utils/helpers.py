"""
src/utils/helpers.py
=====================
Common utility functions used across the Financial Advisor Agent:
  - Rich console formatting / printing
  - Number formatting with Indian currency conventions
  - Percentage formatting with colour
  - Timing decorators
  - Data validation helpers
"""

import time
import logging
import functools
from typing import Any

logger = logging.getLogger(__name__)

# Try to import Rich for pretty printing (optional)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


# ─────────────────────────────────────────────
# Currency / Number Formatting
# ─────────────────────────────────────────────
def format_inr(amount: float, show_sign: bool = False) -> str:
    """
    Format a number as Indian Rupees with lakh/crore notation.
    
    Examples:
        format_inr(150000)   → '₹1.50L'
        format_inr(15000000) → '₹1.50Cr'
        format_inr(5000)     → '₹5,000'
    """
    sign = "+" if (show_sign and amount > 0) else ""

    abs_amount = abs(amount)
    prefix = "-" if amount < 0 else sign

    if abs_amount >= 10_000_000:  # 1 Crore+
        return f"{prefix}₹{abs_amount / 10_000_000:.2f}Cr"
    elif abs_amount >= 100_000:   # 1 Lakh+
        return f"{prefix}₹{abs_amount / 100_000:.2f}L"
    else:
        return f"{prefix}₹{abs_amount:,.0f}"


def format_pct(value: float, show_sign: bool = True, decimals: int = 2) -> str:
    """Format a percentage value."""
    sign = "+" if (show_sign and value > 0) else ""
    return f"{sign}{value:.{decimals}f}%"


def format_change(value: float, is_pct: bool = True) -> str:
    """Format a change value with directional arrow."""
    arrow = "▲" if value > 0 else "▼" if value < 0 else "—"
    if is_pct:
        return f"{arrow} {abs(value):.2f}%"
    return f"{arrow} {format_inr(abs(value))}"


# ─────────────────────────────────────────────
# Console Display Helpers
# ─────────────────────────────────────────────
def print_header(title: str):
    """Print a formatted section header."""
    if RICH_AVAILABLE:
        console.print(Panel(f"[bold cyan]{title}[/bold cyan]", expand=False))
    else:
        width = max(len(title) + 4, 50)
        print("\n" + "═" * width)
        print(f"  {title}")
        print("═" * width)


def print_section(title: str):
    """Print a section divider."""
    if RICH_AVAILABLE:
        console.print(f"\n[bold yellow]── {title} ──[/bold yellow]")
    else:
        print(f"\n── {title} ──")


def print_pnl(label: str, amount: float, pct: float = None):
    """Print a P&L line with colour coding."""
    is_positive = amount >= 0
    amount_str = format_inr(amount, show_sign=True)
    pct_str = f" ({format_pct(pct)})" if pct is not None else ""
    full_str = f"  {label:<25} {amount_str}{pct_str}"

    if RICH_AVAILABLE:
        colour = "green" if is_positive else "red"
        console.print(f"[{colour}]{full_str}[/{colour}]")
    else:
        print(full_str)


def print_success(msg: str):
    if RICH_AVAILABLE:
        console.print(f"[green]✓ {msg}[/green]")
    else:
        print(f"✓ {msg}")


def print_warning(msg: str):
    if RICH_AVAILABLE:
        console.print(f"[yellow]⚠ {msg}[/yellow]")
    else:
        print(f"⚠ {msg}")


def print_error(msg: str):
    if RICH_AVAILABLE:
        console.print(f"[red]✗ {msg}[/red]")
    else:
        print(f"✗ {msg}")


def print_info(msg: str):
    if RICH_AVAILABLE:
        console.print(f"[blue]ℹ {msg}[/blue]")
    else:
        print(f"ℹ {msg}")


def print_holdings_table(holdings: list[dict]):
    """Print portfolio holdings as a formatted table."""
    if RICH_AVAILABLE:
        table = Table(title="Portfolio Holdings", box=box.ROUNDED)
        table.add_column("Symbol", style="cyan", min_width=14)
        table.add_column("Qty", justify="right")
        table.add_column("Avg Buy", justify="right")
        table.add_column("CMP", justify="right")
        table.add_column("Current Value", justify="right")
        table.add_column("Unrealised P&L", justify="right")
        table.add_column("Daily P&L", justify="right")

        for h in holdings:
            pnl_color = "green" if h.get("unrealised_pnl", 0) >= 0 else "red"
            daily_color = "green" if h.get("daily_pnl", 0) >= 0 else "red"
            table.add_row(
                h.get("symbol", ""),
                str(h.get("quantity", "")),
                f"₹{h.get('avg_buy_price', 0):,.1f}",
                f"₹{h.get('current_price', 0):,.1f}",
                format_inr(h.get("current_value", 0)),
                f"[{pnl_color}]{format_inr(h.get('unrealised_pnl', 0), True)} ({format_pct(h.get('unrealised_pnl_pct', 0))})[/{pnl_color}]",
                f"[{daily_color}]{format_inr(h.get('daily_pnl', 0), True)}[/{daily_color}]",
            )
        console.print(table)
    else:
        # Fallback plain text table
        header = f"{'Symbol':<16} {'Qty':>6} {'Avg Buy':>10} {'CMP':>10} {'Value':>12} {'P&L':>14} {'Daily':>12}"
        print("\n" + header)
        print("-" * len(header))
        for h in holdings:
            print(
                f"{h.get('symbol',''):<16} "
                f"{h.get('quantity',''):>6} "
                f"₹{h.get('avg_buy_price',0):>9,.0f} "
                f"₹{h.get('current_price',0):>9,.0f} "
                f"{format_inr(h.get('current_value',0)):>12} "
                f"{format_inr(h.get('unrealised_pnl',0), True):>14} "
                f"{format_inr(h.get('daily_pnl',0), True):>12}"
            )


# ─────────────────────────────────────────────
# Timing Decorator
# ─────────────────────────────────────────────
def timer(func):
    """Decorator to log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug("⏱ %s completed in %.2fs", func.__name__, elapsed)
        return result
    return wrapper


# ─────────────────────────────────────────────
# Validation Helpers
# ─────────────────────────────────────────────
def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division returning default on ZeroDivisionError."""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def truncate(text: str, max_len: int = 100) -> str:
    """Truncate text with ellipsis."""
    return text[:max_len] + "..." if len(text) > max_len else text


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def setup_logging(level: str = "INFO"):
    """Configure application-wide logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
