"""
main.py
========
Entry point for the Autonomous Financial Advisor Agent.

Usage:
  python main.py                          # Run with sample portfolio
  python main.py --portfolio my_port.json # Run with custom portfolio
  python main.py --interactive            # Run + interactive Q&A mode
  python main.py --no-eval               # Skip self-evaluation for speed
"""

import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from src.utils.helpers import setup_logging, print_header, print_error, print_section
from config.settings import LOG_LEVEL


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Autonomous Financial Advisor Agent — Indian Equity Markets"
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        default=None,
        help="Path to portfolio JSON file (default: data/sample_portfolio.json)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive Q&A after analysis",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip self-evaluation step (faster)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="user_001",
        help="User ID for observability tracking (optional)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main():
    """Main application entry point."""
    args = parse_args()

    # ✅ Load env variables
    load_dotenv()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Handle --no-eval flag
    if args.no_eval:
        import os
        os.environ["ENABLE_SELF_EVALUATION"] = "false"

    try:
        # Import agent (after env vars set)
        from src.agent.financial_advisor import AutonomousFinancialAdvisor

        # Initialise agent
        agent = AutonomousFinancialAdvisor(
            portfolio_path=args.portfolio,
        )

        # ✅ FIX: no user_id passed
        result = agent.run()

        if not result.success:
            print_error(f"Agent run failed: {result.error}")
            sys.exit(1)

        # Print full report
        print("\n")
        result.print_full_report()

        # ✅ FIX: Safe observability call
        if hasattr(agent, "print_observability_summary"):
            print("\n")
            agent.print_observability_summary()

        # Interactive Q&A mode
        if args.interactive:
            print_header("Interactive Q&A Mode (type 'quit' to exit)")
            print("You can now ask questions about your portfolio...\n")

            while True:
                try:
                    question = input("Your Question: ").strip()

                    if question.lower() in ("quit", "exit", "q"):
                        print("Goodbye!")
                        break

                    if not question:
                        continue

                    print("\nAnalyzing...")

                    # ✅ SAFE ask()
                    if hasattr(agent, "ask"):
                        answer = agent.ask(result, question)
                        print_section("Advisor Response")
                        print(f"  {answer}\n")
                    else:
                        print("⚠ Interactive mode not supported in this version")

                except KeyboardInterrupt:
                    print("\nSession ended.")
                    break
                except EOFError:
                    break

    except ValueError as e:
        print_error(f"Configuration error: {str(e)}")
        print("\nQuick Setup:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your GROQ_API_KEY")
        print("  3. (Optional) Add Langfuse keys")
        print("  4. Run: python main.py")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)

    except Exception as e:
        logger.exception("Unexpected error in main: %s", str(e))
        print_error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()


print("done")