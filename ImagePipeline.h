/*
   Copyright 2012 Will Sackfield

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

#include <tr1/functional>
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include "tree.hh"

namespace IP
{
	// Had to resort to C callbacks because tr1::function was not into typecasting on the stack
	typedef cv::Mat (*preprocessingFunc)(const cv::Mat,const void*);
	typedef void (*analysisFunc)(const cv::Mat,const void*);
	typedef std::vector<cv::Mat> (*preprocessingSplitFunc)(const cv::Mat,const void*);
	
	cv::Mat downscaleImageBy2(const cv::Mat inputImage,const void* context);
	cv::Mat upscaleImageBy2(const cv::Mat inputImage,const void* context);
	std::vector<cv::Mat> splitChannels(const cv::Mat inputImage,const void* context);
	std::vector<cv::Mat> split11Thresholds(const cv::Mat inputImage,const void* context);
	
	struct NodeFunction
	{
		const std::type_info* type;
		void* function;
	};
	
	struct NodeFunctionRetain : NodeFunction
	{
		int retainCount;
	};
	
	class ImageGraph
	{
		friend class ImagePipeline;
		
		public:
			void addNode(preprocessingFunc func);
			void addNode(analysisFunc func);
			void addNode(preprocessingSplitFunc func);
			void insertNodeAtIndex(preprocessingFunc func,int index);
			void insertNodeAtIndex(analysisFunc func,int index);
			void insertNodeAtIndex(preprocessingSplitFunc func,int index);
			void removeNodeAtIndex(int index);
		private:
			std::vector<NodeFunction> nodes;
	};
	
	class ImagePipeline
	{
		public:
			ImagePipeline(ImageGraph graph);
			
			void addGraph(ImageGraph graph);
			void removeGraph(ImageGraph graph);
			
			void feed(cv::Mat image,void* context=NULL);
		private:
			tree<NodeFunctionRetain> nodes;
			
			static std::vector<tree<NodeFunctionRetain>::iterator> depthTreeRecursion(tree<NodeFunctionRetain>::iterator currentBranch,std::vector<NodeFunction>::iterator currentNode,std::vector<NodeFunction>::iterator endNode);
			static void depthTreeRun(tree<NodeFunctionRetain>::iterator currentBranch,cv::Mat image,void* context);
	};
}
