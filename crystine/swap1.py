# swap1.py
import argparse

def another(ymax,ymin):
    print(ymax , ymin)
    print("Average is: "(ymax+ymin)/2)

def ret_parser():
    parser = argparse.ArgumentParser(
        description="Extracts info from your OUTCAR file")
    parser.add_argument(
        "--ymin", type=float, default=-10.0, help="The first number ymax"
    )
    parser.add_argument(
        "--ymax", type=float, default=4.0, help="The second number ymin"
    )
    return parser


def main():
    # Code here
    print("Swap1 script running!")


    args = ret_parser().parse_args()
    another(
        ymax=args.ymax,
        ymin=args.ymin
    )

if __name__ == "__main__":
    main()
