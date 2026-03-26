# Core Testing

Tests of Core updates. 
- Same environment as for the examples.
- Outputs are piped to `output.log` and `errors.log`.

Test plans are run with a command of the form:
``` bash
python run_plan.py core_test/plans/test_01_plan.json --tool_path core_test/tools --addr bb-1.heph.com:8080 > output.log 2> errors.log
```
The above runs Test 01, modify the plan json file to "02" to run Test 02, and so on.

### Tests with numpy arrays

Test 01 PASS: write 10 samples, read sleep 0. Mean=5.

Test 02 PASS: write 100 samples, read sleep 0. Mean=5.

Test 03 PASS: write 800 samples, read sleep 0. Mean=5.

Test 04 PASS: write 10 samples, multi pass with variable sleeps.

Test 05 PASS: write 100 samples, multi pass with variable sleeps.

Test 06 PASS: write 800 samples, multi pass with variable sleeps.

### Tests of message caching with the add_numpy tool

Test 07 PASS: 1 add agent, 10 samples

Test 08 PASS: 1 add agent, 10 samples, with reverse
- expect samples to have mean 10+sample_num

Test 09 PASS: 1 add agent, 10 samples, with reverse
- expect samples to have mean 10+sample_num

Test 10 PASS: 1 add agent, 800 samples, with reverse
- expect samples to have mean 1000+sample_num

Test 11 PASS: 3 add agents, 10 samples
- expect samples to have mean 2*sample_num+2
- last post should be add_numpy_3, sample 9, mean 20

Test 12 PASS: 3 add agents, 10 samples, with reverse
- expect samples to have mean 2*sample_num+2
- last output should be add_numpy_3, sample 9, mean 20

Test 13 PASS: 3 add agents, 10 samples, with reverse
- expect samples to have mean 2*sample_num+2
- last post should be add_numpy_3, sample 0, mean 2

Test 14 PASS: 3 add agents, 800 samples, with reverse
- expect samples to have mean 2*sample_num+2
- last output should be add_numpy_3, sample 799, mean 1600

Test 15 PASS: 3 add agents, 10 samples
- last warning should be add_numpy_1, sample 9, None (since array dimensions do not match)
- last output should be add_numpy_3, sample 9, None

Test 16 PASS: 3 add agents, 10 samples, with reverse
- last warning should be add_numpy_1, sample 0, None (since array dimensions do not match)
- last output should be add_numpy_3, sample 0, None

