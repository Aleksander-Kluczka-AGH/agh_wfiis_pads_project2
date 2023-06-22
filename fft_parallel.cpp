#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

#include <upcxx/upcxx.hpp>

#define ND [[nodiscard]]

namespace global
{
	
    int process_count;
    int slave_count;
    int rank;

    std::vector<float> input;
    int input_size;

}  // namespace global

namespace logger
{

void all(const char*, auto...);
void master(const char*, auto...);
void slave(const char*, auto...);

}  // namespace logger

#ifdef ENABLE_LOGGING
	#define LOG_ALL logger::all
	#define LOG_MASTER logger::master
	#define LOG_SLAVE logger::slave
#else
	#define LOG_ALL(...)
	#define LOG_MASTER(...)
	#define LOG_SLAVE(...)
#endif

void initGlobals();
ND int reverseBits(int, int);
ND int getProcessCount();
ND int getProcessRank();
void initInputValues(const char*);
void showResults(const float*, const float*, double);

int main(int argc, char** argv) {
	// Initialize UPC++, global variables and input sequence.
    upcxx::init();
    initGlobals();
    initInputValues("res/input.txt");
    
	// Create pointers for shared memory
	upcxx::global_ptr<float> seq_real = nullptr;
	upcxx::global_ptr<float> seq_img = nullptr;
	
	if(global::rank == 0)
	{
		// Allocate shared memory
		seq_real = upcxx::new_array<float>(global::input_size);
		seq_img = upcxx::new_array<float>(global::input_size);
		
		// Split input into real and imaginary sequences.
		// Apply bit reversing for butterfly indexing algorithm.
		const int max_bit_width = std::log2f(global::input_size);
		for (int i = 0; i < global::input_size; i++)
		{
			upcxx::rput(global::input[reverseBits(i - 1, max_bit_width) + 1], seq_real + i).wait();
			upcxx::rput(0.0f, seq_img + i).wait();
		}
	}

	// Broadcast shared memory from master to slaves
	seq_real = upcxx::broadcast(seq_real, 0).wait();
	seq_img = upcxx::broadcast(seq_img, 0).wait();

    const int values_per_process = global::input_size / global::slave_count;

	// Begin FFT computation with butterfly indexing applied.
    auto starttime = std::chrono::high_resolution_clock::now();
	for(int div = 1, key = std::log2f(global::input_size - 1); key > 0; key--, div *= 2)
    {
        LOG_MASTER("ITERATION %d\n", static_cast<int>(std::log2f(div)));
		float temp_real[values_per_process]{};
		float temp_img[values_per_process]{};
        if(global::rank != 0)
        {
            // Compute more than one FFT node on each worker if (number of nodes > number of
            // workers). If there are less nodes than workers, the behaviour is undefined.
            LOG_SLAVE("beginning compute...\n");
            for(int b = 0; b < values_per_process; b++)
            {
                // Prepare butterfly indexing.
                const auto b_rank = (global::rank - 1) * values_per_process + b + 1;
                const auto is_even = ((b_rank + div - 1) / div) % 2;
                const auto is_odd = 1 - is_even;
                const auto butterfly_index = M_PI * ((b_rank - 1) % (div * 2)) / div;

                // Calculate FFT sequence on each worker in parallel.
				float seq_real_odd = upcxx::rget(seq_real + b_rank - (div * is_odd)).wait();
				float seq_real_even = upcxx::rget(seq_real + b_rank + (div * is_even)).wait();
				float seq_img_odd = upcxx::rget(seq_img + b_rank - (div * is_odd)).wait();
				float seq_img_even = upcxx::rget(seq_img + b_rank + (div * is_even)).wait();
				
                temp_real[b] =
                    seq_real_odd
                    + (std::cos(butterfly_index) * seq_real_even)
                    + (std::sin(butterfly_index) * seq_img_even);

                temp_img[b] =
                    seq_img_odd
                    + (std::cos(butterfly_index) * seq_img_even)
                    - (std::sin(butterfly_index) * seq_real_even);
            }
        }
		
		// Wait for all computations to finish before putting results into shared memory.
		upcxx::barrier();
		
		// Put current just calculated values from each worker into shared memory.
		for(int b = 0; b < values_per_process; b++)
		{
			const auto b_rank = (global::rank - 1) * values_per_process + b + 1;
			upcxx::rput(temp_real[b], seq_real + b_rank).wait();
			upcxx::rput(temp_img[b], seq_img + b_rank).wait();
		}
		LOG_SLAVE("ending compute...\n");

        // Wait for all workers to put their computations into shared memory.
        upcxx::barrier();
    }
	auto endtime = std::chrono::high_resolution_clock::now();

	// Print computed FFT results.
	if(global::rank == 0)
	{
		double duration = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count();
		float* seq_real_arr = seq_real.local();
		float* seq_img_arr = seq_img.local();
		showResults(seq_real_arr, seq_img_arr, duration);
	}

	// Clean up UPC++ internal state and exit the program.
    upcxx::finalize();
    return 0;
}

////////////////

/**
 * @brief Initialize global variables set by UPC++.
 *
 */
void initGlobals()
{
    global::process_count = getProcessCount();
    global::slave_count = global::process_count - 1;
    global::rank = getProcessRank();
}

/**
 * @brief Reverse bits to reverse recursive butterfly indexing algorithm.
 *
 * @param number Index of a given butterfly node of type int32.
 * @param bit_range Range of least significant bits to reverse in the number parameter.
 * @return int Index of a node with reversed bit_range least significant bits.
 */
int reverseBits(int number, int bit_range)
{
    int reverse_number = 0;
    for(int i = 0; i < bit_range; i++)
    {
        reverse_number |= ((number >> i) & 1) << (bit_range - 1 - i);
    }
    return reverse_number;
}

/**
 * @brief Read the process count from UPC++.
 *
 * @return int Number of parallely running processes.
 */
int getProcessCount()
{
    return upcxx::rank_n();
}

/**
 * @brief Obtain current process rank. Master receives rank 0, meanwhile workers acquire rank > 0.
 *
 * Ranks are unique, meaning that no two processes can be labeled with the same rank.
 *
 * @return int Rank of the current process.
 */
int getProcessRank()
{
    return upcxx::rank_me();
}

/**
 * @brief Initialize all processes with input values read by master.
 *
 * Flow:
 * 1) Read from file on master process.
 * 2) Broadcast the size of input to all workers.
 * 3) Declare array of obtained size on each process.
 * 4) Broadcast input sequence from master to all workers.
 *
 * @param path Directory path to a text file with an array of input values.
 */
void initInputValues(const char* path)
{
	// Prepare variables to read input data from master
	upcxx::global_ptr<float> input_ptr = nullptr;
	upcxx::global_ptr<int> input_size_ptr = nullptr;

    // Read from input file on master.
    if(global::rank == 0)
    {
		// Read file into a temp vector
		std::vector<float> temp;

        std::ifstream file(path);

        if(not file.is_open())
        {
			std::printf("Couldn't open file %s\n", path);
            return;
        }

        temp.push_back(0);

        float real{};
        while(file >> real)
        {
            temp.push_back(real);
        }

		// Allocate shared memory
		input_ptr = upcxx::new_array<float>(temp.size());
		input_size_ptr = upcxx::new_<int>();
		
		// Put data from temp vector into shared memory
		for(int i = 0; i < temp.size(); i++)
		{
			upcxx::rput(temp[i], input_ptr + i).wait();
		}
		
		// Put input size into shared memory
		int size_int = temp.size();
		upcxx:rput(size_int, input_size_ptr).wait();
    }

	// Broadcast shared memory from master to slaves
	input_ptr = upcxx::broadcast(input_ptr, 0).wait();
	input_size_ptr = upcxx::broadcast(input_size_ptr, 0).wait();

	// Retrieve input size from shared memory and assign into "global" variable (stored locally)
	global::input_size = upcxx::rget(input_size_ptr).wait();

	// Retrieve input from shared memory and push into "global" input vector (stored locally)
	for(int i = 0; i < global::input_size; i++)
	{
		global::input.push_back(upcxx::rget(input_ptr + i).wait());
	}
}

/**
 * @brief Print results of computation.
 *
 * @param seq_real Sequence of results in real domain.
 * @param seq_img Sequence of results in imaginary domain.
 * @param starttime Timestamp of algorithm before computation.
 * @param endtime Timestamp of algorithm after computation has finished.
 */
void showResults(const float* seq_real, const float* seq_img, double duration)
{
    // Printing only on master process.
    if(0 == global::rank)
    {
        std::printf("\n");
        for(int i = 1; i < global::input_size; i++)
        {
            const char img_sign = seq_img[i] >= 0 ? '+' : '-';
            std::printf("X[%3d] = %6.2f %c i%-6.2f\n",
                i,
                seq_real[i],
                img_sign,
                std::fabs(seq_img[i]));
        }

        std::printf("\nParallel FFT computation time: %.4lf ms\n\n", duration);
    }
}

namespace logger
{

/**
 * @brief Print logging message on every process.
 *
 * @param format Format string provided to std::printf().
 * @param args Formatting parameters provided to std::printf().
 */
void all(const char* format, auto... args)
{
    const auto rank_idx = not not global::rank;

    std::array<std::string, 2> thread_name = {"master",
        "slave(" + std::to_string(global::rank) + ")"};

    std::stringstream strstr;
    strstr << "LOG | " << thread_name[rank_idx] << " | " << format;
    std::printf(strstr.str().c_str(), args...);
}

/**
 * @brief Print logging message only on master process.
 *
 * @param format Format string provided to std::printf().
 * @param args Formatting parameters provided to std::printf().
 */
void master(const char* format, auto... args)
{
    if(global::rank == 0)
    {
        logger::all(format, args...);
    }
}

/**
 * @brief Print logging message only on worker processes (effectively excluding master).
 *
 * @param format Format string provided to std::printf().
 * @param args Formatting parameters provided to std::printf().
 */
void slave(const char* format, auto... args)
{
    if(global::rank != 0)
    {
        ::logger::all(format, args...);
    }
}

}  // namespace logger
