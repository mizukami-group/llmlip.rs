#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "../src/llmlip_openmm_capi.cpp"

extern "C" {
llmlip_status_t llmlip_model_load(const char *, llmlip_model_t **) {
  return LLMLIP_ERROR_MODEL_LOAD;
}

llmlip_status_t llmlip_model_load_baked_mace_oam_medium(llmlip_model_t **) {
  return LLMLIP_ERROR_UNSUPPORTED;
}

void llmlip_model_free(llmlip_model_t *) {}

llmlip_status_t llmlip_model_capabilities(const llmlip_model_t *,
                                          llmlip_model_capabilities_t *) {
  return LLMLIP_ERROR_INVALID_ARGUMENT;
}

llmlip_status_t llmlip_predict_external_graph(const llmlip_model_t *,
                                              const llmlip_structure_view_t *,
                                              const llmlip_external_graph_view_t *,
                                              const llmlip_prediction_options_t *,
                                              llmlip_prediction_out_t *) {
  return LLMLIP_ERROR_INVALID_ARGUMENT;
}

llmlip_status_t llmlip_predict_structure(const llmlip_model_t *,
                                         const llmlip_structure_view_t *,
                                         const llmlip_prediction_options_t *,
                                         llmlip_prediction_out_t *) {
  return LLMLIP_ERROR_INVALID_ARGUMENT;
}

const char *llmlip_last_error(void) { return "stub"; }
void llmlip_clear_error(void) {}
}

namespace {

void require_near(double actual, double expected, const char *label) {
  if (std::fabs(actual - expected) > 1.0e-12) {
    std::cerr << label << ": expected " << expected << ", got " << actual << "\n";
    std::exit(1);
  }
}

std::array<double, 3> fractional_to_cartesian(const std::array<double, 3> &frac,
                                              const std::array<double, 9> &cell) {
  return {
      frac[0] * cell[0] + frac[1] * cell[3] + frac[2] * cell[6],
      frac[0] * cell[1] + frac[1] * cell[4] + frac[2] * cell[7],
      frac[0] * cell[2] + frac[1] * cell[5] + frac[2] * cell[8],
  };
}

void test_wraps_orthorhombic_positions() {
  const std::array<double, 9> cell{10.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 30.0};
  std::vector<double> positions{12.0, -5.0, 75.0};

  llmlip_openmm::wrap_positions_into_periodic_cell(positions, cell);

  require_near(positions[0], 2.0, "orthorhombic x");
  require_near(positions[1], 15.0, "orthorhombic y");
  require_near(positions[2], 15.0, "orthorhombic z");
}

void test_wraps_triclinic_positions() {
  const std::array<double, 9> cell{
      2.0, 0.1, 0.2,
      0.5, 3.0, 0.3,
      0.4, 0.25, 4.0,
  };
  const std::array<double, 3> fractional{1.25, -0.5, 2.125};
  const std::array<double, 3> expected_fractional{0.25, 0.5, 0.125};
  const std::array<double, 3> cartesian = fractional_to_cartesian(fractional, cell);
  const std::array<double, 3> expected = fractional_to_cartesian(expected_fractional, cell);
  std::vector<double> positions{cartesian[0], cartesian[1], cartesian[2]};

  llmlip_openmm::wrap_positions_into_periodic_cell(positions, cell);

  require_near(positions[0], expected[0], "triclinic x");
  require_near(positions[1], expected[1], "triclinic y");
  require_near(positions[2], expected[2], "triclinic z");
}

void test_singular_cell_fails() {
  const std::array<double, 9> cell{1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0};
  std::vector<double> positions{0.0, 0.0, 0.0};
  try {
    llmlip_openmm::wrap_positions_into_periodic_cell(positions, cell);
  } catch (const llmlip_openmm::Error &error) {
    const std::string message = error.what();
    if (message.find("singular") != std::string::npos) return;
    std::cerr << "unexpected singular-cell error: " << message << "\n";
    std::exit(1);
  }
  std::cerr << "singular cell did not fail\n";
  std::exit(1);
}

}  // namespace

int main() {
  try {
    test_wraps_orthorhombic_positions();
    test_wraps_triclinic_positions();
    test_singular_cell_fails();
  } catch (const std::exception &error) {
    std::cerr << "unexpected exception: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
