.PHONY: runtest clean

runtest: test_main.cpp
	$(CXX) -std=c++17 test_main.cpp -lgtest -lgtest_main -L/opt/anaconda3/envs/reneu/lib -o xiuli_unit_tests -O0


clean:
	rm -f xiuli_unit_tests 