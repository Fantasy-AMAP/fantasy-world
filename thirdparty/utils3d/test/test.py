import importlib
import os
import torch
import traceback

CRED    = '\033[91m'
CGREEN  = '\033[92m'
CEND    = '\033[0m'

if __name__ == '__main__':
    # list all tests
    tests = []
    for root, dirs, files in os.walk('test'):
        if root == 'test':
            continue
        for file in files:
            if file.endswith('.py'):
                root = root.replace('test/', '').replace('test\\', '')
                test = os.path.join(root, file)
                test = test.replace('/', '.').replace('\\', '.').replace('.py', '')
                tests.append(test)
    tests.sort()
    print(f'Found {len(tests)} tests:')
    for test in tests:
        print(f'  {test}')
    print()

    # disable torch optimizations
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # import and run
    passed = 0
    for test in tests:
        print(f'Running test: {test}... ', end='')
        test = importlib.import_module(test, '.'.join(test.split('.')[:-1]))
        try:
            test.run()
        except Exception as e:
            print(CRED, end='')
            print('Failed')
            traceback.print_exc()
        else:
            print(CGREEN, end='')
            print('Passed')
            passed += 1
        print(CEND, end='')

    print(f'Passed {passed}/{len(tests)} tests')
    