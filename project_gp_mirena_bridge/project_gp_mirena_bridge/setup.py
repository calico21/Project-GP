"""
setup.py — ROS2 Python package for project_gp_mirena_bridge.

Place this in the root of your project_gp ROS2 package alongside package.xml.
The bridge node is then callable as:
    ros2 run project_gp mirena_bridge
"""

from setuptools import setup, find_packages

package_name = "project_gp"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "jax",
        "jaxlib",
        "optax",
        "numpy",
    ],
    zip_safe=True,
    maintainer="Tecnun eRacing",
    maintainer_email="eracing@tecnun.es",
    description="Project-GP ↔ MirenaSim ROS2 shadow controller bridge",
    license="GPL-3.0",
    entry_points={
        "console_scripts": [
            # ros2 run project_gp mirena_bridge
            "mirena_bridge = project_gp_mirena_bridge.mirena_bridge:main",
        ],
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# package.xml (place in ROS2 package root — copy to XML file):
# ─────────────────────────────────────────────────────────────────────────────
PACKAGE_XML = """<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>project_gp</name>
  <version>0.1.0</version>
  <description>Project-GP differentiable digital twin — MirenaSim bridge</description>
  <maintainer email="eracing@tecnun.es">Tecnun eRacing</maintainer>
  <license>GPL-3.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>mirena_common</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
"""
