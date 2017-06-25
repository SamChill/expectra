
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trajectories', metavar='TRAJ', nargs='+',
            help='trajectory file (POSCAR, con, xyz)')
    args = parser.parse_args()
    for filename in args.trajectories:
        print(filename[-3:])
if __name__ == '__main__':
    main()

