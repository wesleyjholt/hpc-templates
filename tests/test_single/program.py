print(__name__)
print('hello world')

if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--foo', type=str)
    parser.add_argument('--bar', type=float)
    args = parser.parse_args()

    print('foo is a variable of type: ', type(args.foo))
    print('The value of foo is: ', args.foo)
    print('bar is a variable of type: ', type(args.bar))
    print('The value of bar + 1 is: ', args.bar + 1)
