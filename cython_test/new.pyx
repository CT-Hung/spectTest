cdef extern from "./detection_algorithm.cpp":
    unsigned int frequency_mapping(unsigned int input_index, int fs, int N)
def spect(input_index, fs, N):
    return frequency_mapping(input_index, fs, N)
