#!/usr/bin/env python

import multiprocessing
import time

def test():
    try:
        time.sleep(1)
        a = 1/0
    except Exception as e:
        print(e)
    return


def main():
    pool_size = int(multiprocessing.cpu_count() * 0.8)
    pool = multiprocessing.Pool(processes=pool_size)

    for i in range(10):
        print('multiprocessing -- ', i)
        pool.apply_async(test,)
    pool.close()
    pool.join()
    print('multiprocessing all done', '\n')


    print(time.time())


if __name__ == '__main__':
    main()