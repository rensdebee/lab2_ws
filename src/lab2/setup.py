from setuptools import find_packages, setup

package_name = "lab2"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rens",
    maintainer_email="r.35rens@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "lab2 = lab2.lab2:main",
            "lab2_mt = lab2.lab2_mt:main",
            "position = lab2.position:main",
        ],
    },
)
