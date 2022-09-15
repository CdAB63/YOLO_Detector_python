# YOLO_Detector_python
A python yolo detector class build upon the darknet.py in order to encapsulate functionalities. FIxed the memory leak issue.

It does not include libdarknet.so because it is dependent on the architecture (x96_64, AARCH64, linux, OS10, Windows, etc) and also because it would freeze the library in a given version.

To build the libdarknet.so just:
1. git clone https://github.com/AlexeyAB/darknet
2. cd darknet
3. edit the Makefile (if not Windows)
4. make

And there will be the libdarknet.so in the build directory

TO build the windows version, just follow the instructions in AlexeyAB git

Thank you
