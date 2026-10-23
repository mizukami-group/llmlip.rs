#include "llmlip_openmm_capi.h"

#include <exception>
#include <iostream>

int main() {
  try {
    auto model = llmlip_openmm::Model::loadBakedMaceOamMedium();
    (void)model.capabilities();
    return 0;
  } catch (const llmlip_openmm::Error &error) {
    (void)error;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "unexpected exception: " << error.what() << "\n";
    return 1;
  }

  return 0;
}
