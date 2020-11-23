.PHONY: runtest clean

runtest: test_segmentation.cpp
	$(CXX) -std=c++17 test_segmentation.cpp -I/Users/jwu/opt/anaconda3/envs/reneu/include -L/Users/jwu/opt/anaconda3/envs/reneu/lib -lgtest -lgtest_main -L/opt/anaconda3/envs/reneu/lib -o tests -O0


clean:
	rm -f xiuli_unit_tests 