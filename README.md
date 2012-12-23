ImagePipeline 1.0
==============

ImagePipeline is a C++ API for designing OpenCV solutions through the use of pipelines rather than a set of specific functions. The inspiration came from spending lots of time meshing methods together to obtain efficient solutions preventing the use of repeating preprocessing steps, by using this API I can simply share the pipeline and when adding my ImageGraph (a set of preprocessing steps) to the pipeline it will make a tree representing the most efficient way of combining the graphs.

The example contains an implementation of the squares OpenCV demo program written in this way.

Distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0 "Apache License, Version 2.0")
Compiled and tested on OSX

Third party code
----------------

- [tree.hh](http://tree.phi-sci.com "tree.hh: an STL-like C++ tree class")

Dependencies
------------

- [opencv](http://opencv.willowgarage.com/wiki/ "OpenCV")
