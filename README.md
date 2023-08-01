# Parallel and distributed systems - Project 2

Project written for Applied Computer Science **Parallel and distributed systems** course at [AGH UST](https://www.agh.edu.pl/en) [WFiIS](https://www.fis.agh.edu.pl/en/).

Parallel FFT computation using UPC++ and C++.

Semester `8`, year `2023`.

## How to run

The program is designed to work on an AGH laboratory machine, it will not work in different
environment.

```bash
$ mkdir build
$ cd build
$ cmake ..

$ make run         # run parallel FFT computation
$ make sequential  # run sequential FFT computation
$ make clean       # clean build objects
```

### Additional parameters

CMake file allows specifying extra flags listed below:

| Flag             | Description                                                                       |
| ---------------- | --------------------------------------------------------------------------------- |
| `ENABLE_LOGGING` | Print debug logs in parallel algorithm.                                           |
| `ENABLE_ASAN`    | Enable address sanitizer (useful when debugging parallel code on single process). |

They can be used as shown below:

```bash
$ cmake -DENABLE_LOGGING=1 -DENABLE_ASAN=1 ..
```

## Authors

- [Piotr Deda](https://github.com/PiotrDeda)
- [Aleksander Kluczka](https://github.com/vis4rd)


## License

Sources are licensed under MIT, a free and open-source license. For details, please see [the license file](LICENSE.md).
