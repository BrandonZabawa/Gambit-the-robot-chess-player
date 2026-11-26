// Python bindings (built only with -DGAMBIT_BUILD_PYTHON=ON). Lets the notebooks compare the
// Python and C++ paths:   import gambit_cpp_py as g;  g.ping_packet(1); g.forward([0]*5)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gambit/inference/onnx_classifier.hpp"
#include "gambit/kinematics/so101.hpp"
#include "gambit/servo_bus/sts_protocol.hpp"
#include "gambit/square_map/square_map.hpp"

namespace py = pybind11;
namespace gs = gambit::servo;
namespace gk = gambit::kinematics;
namespace gq = gambit::square_map;

PYBIND11_MODULE(gambit_cpp_py, m) {
  m.doc() = "Gambit C++ layer (servo protocol, kinematics, square map)";

  m.def("checksum", [](const std::vector<std::uint8_t>& body) { return gs::checksum(body); });
  m.def("ping_packet", &gs::ping);
  m.def("read_packet", &gs::read);
  m.def("write_packet", [](std::uint8_t id, std::uint8_t addr, const std::vector<std::uint8_t>& d) {
    return gs::write(id, addr, d);
  });
  m.def("ticks_to_rad", &gs::ticks_to_rad);
  m.def("rad_to_ticks", &gs::rad_to_ticks);

  m.def("forward", [](const std::vector<double>& q) {
    if (q.size() != gk::kNumJoints) throw std::invalid_argument("need 5 joint angles");
    gk::JointVector v;
    for (int i = 0; i < gk::kNumJoints; ++i) v[i] = q[i];
    const Eigen::Matrix4d t = gk::forward(v).matrix();
    std::vector<std::vector<double>> out(4, std::vector<double>(4));
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c) out[r][c] = t(r, c);
    return out;
  }, "4x4 tool pose as nested lists");

  m.def("square_to_world", [](const std::string& sq) {
    auto p = gq::square_to_world(sq);
    if (!p) throw std::invalid_argument("bad square");
    return std::vector<double>{p->x, p->y, p->z};
  });
  m.def("world_to_square", [](double x, double y) { return gq::world_to_square(x, y); });

  m.def("onnx_available", &gambit::inference::OnnxClassifier::available);
  // TODO: bind SerialPort and OnnxClassifier (numpy buffers via pybind11/numpy.h) once used.
}
