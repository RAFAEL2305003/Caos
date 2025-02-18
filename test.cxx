#include <iostream>
#include <fstream>
#include <cmath>
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

	size_t n = 1, c = 3, w, h;
	
	std::ifstream image("image.ppm");
	// image >> std::endl;	
	std::string str;
	image >> str;
	// image >> std::endl;
	float f;
	image >> str;
	std::cout << f << std::endl;
	// auto src_md = dnnl::memory::desc(
    //        {n, c, h, w}, // logical dims, the order is defined by a primitive
    //        dnnl::memory::data_type::f32, // tensor's data type
    //        dnnl::memory::format_tag::nhwc // memory format, NHWC in this case
    // );	

	return 0;
}
