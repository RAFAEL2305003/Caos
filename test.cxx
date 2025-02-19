#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include "/home/rafaelrodrigues/oneDNN/oneAPI-samples/Libraries/oneDNN/getting_started/build/src/example_utils.hpp"
#include <oneapi/dnnl/dnnl.hpp>

int main() {
	dnnl::engine eng(dnnl::engine::kind::cpu, 0);
	dnnl::stream engine_stream(eng);
	
	// std::ofstream image("image.ppm");
	// image << "P3" << std::endl;
	// image << "64 64" << std::endl;
	// image << "255" << std::endl;
	// for(size_t i = 0; i < 64; i++) {
	// 	for(size_t j = 0; j < 64; j++) {
	// 		int k = 0;
	// 		while(k < 3) {
	// 			image << i + j + k << " ";
	// 			k++;
	// 		}
	// 		image << std::endl;
	// 	}
	// }
	// image.close();

	int n = 1, c = 3, w, h;
	
	std::ifstream image("image.ppm");
	std::string str;
	if(!image.is_open()) {
		std::cerr << "Error opening the file!";
		return 1;
	}
	
	std::string f;
	getline(image, f);	
	image >> w;
	image >> h;	
	getline(image, f);
	getline(image, f);
	std::vector<int> v_image(h * w * c * n);
	size_t i = 0;
	char del = ' ';
	while(getline(image, f, del)) {
		int x = atoi(f.c_str());
		v_image[i] = x;
		i++;
	}

	auto src_md = dnnl::memory::desc(
           {n, c, h, w}, // logical dims, the order is defined by a primitive
           dnnl::memory::data_type::f32, // tensor's data type
           dnnl::memory::format_tag::nhwc // memory format, NHWC in this case
    );	
	
	auto src_mem = dnnl::memory(src_md, eng);
	write_to_dnnl_memory(v_image.data(), src_mem);
	auto dst_mem = dnnl::memory(src_md, eng);

	auto relu_pd = dnnl::eltwise_forward::primitive_desc(
        eng, // an engine the primitive will be created for
        dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_relu,
        src_md, // source memory descriptor for an operation to work on
        src_md, // destination memory descriptor for an operation to work on
        0.f, // alpha parameter means negative slope in case of ReLU
        0.f // beta parameter is ignored in case of ReLU
	);

	// ReLU primitive
	auto relu = dnnl::eltwise_forward(relu_pd); // !!! this can take quite some time
	// Execute ReLU (out-of-place)
	relu.execute(engine_stream, // The execution stream
			{
					// A map with all inputs and outputs
					{DNNL_ARG_SRC, src_mem}, // Source tag and memory obj
					{DNNL_ARG_DST, dst_mem}, // Destination tag and memory obj
			});

	// Wait the stream to complete the execution
	engine_stream.wait();

	std::vector<float> relu_image(n * c * h * w);
	read_from_dnnl_memory(relu_image.data(), dst_mem);
	i = 0;
	// Check the results
	for (int j = 0; j < n; ++j)
		for (int k = 0; k < h; ++k)
			for (int l = 0; l < w; ++l)
				for (int m = 0; m < c; ++m) {
					float expected = v_image[i] < 0
							? 0.f
						:v_image[i]; // expected value
					if (relu_image[i] != expected) {
						std::cout << "At index(" << j << ", " << m << ", " << h
								  << ", " << k << ") expect " << expected
								  << " but got " << relu_image[i]
								  << std::endl;
						throw std::logic_error("Accuracy check failed.");
					}
           }

	image.close();

    std::cout << "Example" << " on "
              << engine_kind2str_upper(dnnl::engine::kind::cpu) << "." << std::endl;
	return 0;
}
