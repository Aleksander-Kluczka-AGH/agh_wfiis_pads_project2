#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>

namespace global
{

std::vector<float> input;
int input_size{};

}  // namespace global

void initInputValues(const char*);
void showResults(const std::vector<float>&, const std::vector<float>&, std::clock_t, std::clock_t);

int main(int argc, char** argv)
{
    initInputValues("../res/input.txt");

    std::vector<float> result_real(global::input_size);
    std::vector<float> result_img(global::input_size);

    std::clock_t starttime = std::clock();

    for(int res = 0; res < global::input_size; res++)
    {
        result_real[res] = 0.0;
        result_img[res] = 0.0;

        for(int inp = 0; inp < global::input_size; inp++)
        {
            const float temp = -2 * M_PI * inp * res / global::input_size;
            result_real[res] += (global::input[inp] * std::cos(temp));
            result_img[res] += (global::input[inp] * std::sin(temp));
        }
    }

    std::clock_t endtime = std::clock();
    showResults(result_real, result_img, starttime, endtime);

    return 0;
}

void initInputValues(const char* path)
{
    std::ifstream file(path);
    if(not file.is_open())
    {
        return;
    }

    float temp{};
    while(file >> temp)
    {
        global::input.push_back(temp);
    }
    global::input_size = global::input.size();
}

void showResults(const std::vector<float>& result_real,
    const std::vector<float>& result_img,
    std::clock_t starttime,
    std::clock_t endtime)
{
    std::printf("\n");
    for(int result_iter = 0; result_iter < global::input_size; result_iter++)
    {
        const char img_sign = result_img[result_iter] >= 0 ? '+' : '-';
        std::printf("X[%3d] = %6.2f %c i%-6.2f\n",
            result_iter,
            result_real[result_iter],
            img_sign,
            std::fabs(result_img[result_iter]));
    }
    std::printf("\n");

    double cpu_time_used = (static_cast<double>(endtime - starttime)) / CLOCKS_PER_SEC;
    std::printf("\nSequential FFT computation time: %.4lf ms\n\n", cpu_time_used * 1000);
}
