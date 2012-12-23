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

#include "ImagePipeline.h"

namespace IP
{
	cv::Mat downscaleImageBy2(const cv::Mat inputImage,const void* context)
	{
		cv::Mat outputImage(cv::Size(inputImage.size().width/2,inputImage.size().height/2),inputImage.type());
		cv::pyrDown(inputImage,outputImage,outputImage.size());
		return outputImage;
	}
	
	cv::Mat upscaleImageBy2(const cv::Mat inputImage,const void* context)
	{
		cv::Mat outputImage(cv::Size(inputImage.size().width*2,inputImage.size().height*2),inputImage.type());
		cv::pyrUp(inputImage,outputImage,outputImage.size());
		return outputImage;
	}
	
	std::vector<cv::Mat> splitChannels(const cv::Mat inputImage,const void* context)
	{
		std::vector<cv::Mat> outputImages;
		for(int i=0;i<inputImage.channels();i++)
		{
			cv::Mat grayImage(inputImage.size(),CV_8U);
			const int ch[] = { i, 0 };
			mixChannels(&inputImage,1,&grayImage,1,ch,1);
			outputImages.push_back(grayImage);
		}
		return outputImages;
	}
	
	std::vector<cv::Mat> split11Thresholds(const cv::Mat inputImage,const void* context)
	{
		std::vector<cv::Mat> outputImages;
		for(int i=0;i<11;i++)
		{
			cv::Mat outputImage(inputImage.size(),CV_8U);
			if(i == 0)
			{
				cv::Canny(inputImage,outputImage,0,50,5);
				cv::dilate(outputImage,outputImage,cv::Mat());
			}
			else
				outputImage = inputImage >= ((i+1)*255)/11;
			outputImages.push_back(outputImage);
		}
		return outputImages;
	}
	
	void ImageGraph::addNode(preprocessingFunc func)
	{
		NodeFunction nodeFunction = { &typeid(func), (void*)func };
		nodes.push_back(nodeFunction);
	}
	
	void ImageGraph::addNode(analysisFunc func)
	{
		NodeFunction nodeFunction = { &typeid(func), (void*)func };
		nodes.push_back(nodeFunction);
	}
	
	void ImageGraph::addNode(preprocessingSplitFunc func)
	{
		NodeFunction nodeFunction = { &typeid(func), (void*)func };
		nodes.push_back(nodeFunction);
	}
	
	void ImageGraph::insertNodeAtIndex(preprocessingFunc func,int index)
	{
		NodeFunction nodeFunction = { &typeid(func), (void*)func };
		nodes.insert(nodes.begin()+index,nodeFunction);
	}
	
	void ImageGraph::insertNodeAtIndex(analysisFunc func,int index)
	{
		NodeFunction nodeFunction = { &typeid(func), (void*)func };
		nodes.insert(nodes.begin()+index,nodeFunction);
	}
	
	void ImageGraph::insertNodeAtIndex(preprocessingSplitFunc func,int index)
	{
		NodeFunction nodeFunction = { &typeid(func), (void*)func };
		nodes.insert(nodes.begin()+index,nodeFunction);
	}
	
	void ImageGraph::removeNodeAtIndex(int index)
	{
		nodes.erase(nodes.begin()+index);
	}
	
	ImagePipeline::ImagePipeline(ImageGraph graph)
	{
		this->addGraph(graph);
	}
	
	void ImagePipeline::addGraph(ImageGraph graph)
	{
		std::vector<NodeFunction>::iterator currentNode = graph.nodes.begin();
		tree<NodeFunctionRetain>::sibling_iterator currentIterator = nodes.begin();
		tree<NodeFunctionRetain>::sibling_iterator endIterator = nodes.end();
		
		std::vector<tree<NodeFunctionRetain>::iterator> maxPath;
		
		while(currentIterator != endIterator)
		{
			NodeFunction nodeFunction = *currentNode;
			NodeFunctionRetain treeFunction = *currentIterator;
			if(nodeFunction.function == treeFunction.function)
			{
				std::vector<tree<NodeFunctionRetain>::iterator> path = ImagePipeline::depthTreeRecursion(currentIterator,currentNode+1,graph.nodes.end());
				if(path.size()+1 > maxPath.size())
				{
					maxPath.clear();
					maxPath.push_back(currentIterator);
					for(int i=0;i<path.size();i++)
						maxPath.push_back(path[i]);
				}
			}
			currentIterator++;
		}
		
		tree<NodeFunctionRetain>::iterator pathIterator;
		for(int i=0;i<graph.nodes.size();i++)
		{
			if(maxPath.size()-1 <= i)
			{
				pathIterator = maxPath[i];
				(*pathIterator).retainCount++;
			}
			else
			{
				NodeFunctionRetain nodeFunction;
				nodeFunction.type = graph.nodes[i].type;
				nodeFunction.function = graph.nodes[i].function;
				nodeFunction.retainCount = 1;
				if(i == 0)
					pathIterator = nodes.insert(nodes.begin(),nodeFunction);
				else
					pathIterator = nodes.append_child(pathIterator,nodeFunction);
			}
		}
	}
	
	void ImagePipeline::removeGraph(ImageGraph graph)
	{
		std::vector<NodeFunction>::iterator currentNode = graph.nodes.begin();
		tree<NodeFunctionRetain>::sibling_iterator currentIterator = nodes.begin();
		tree<NodeFunctionRetain>::sibling_iterator endIterator = nodes.end();
		
		std::vector<tree<NodeFunctionRetain>::iterator> maxPath;
		
		while(currentIterator != endIterator)
		{
			NodeFunction nodeFunction = *currentNode;
			NodeFunctionRetain treeFunction = *currentIterator;
			if(nodeFunction.function == treeFunction.function)
			{
				std::vector<tree<NodeFunctionRetain>::iterator> path = ImagePipeline::depthTreeRecursion(currentIterator,currentNode+1,graph.nodes.end());
				if(path.size()+1 > maxPath.size())
				{
					maxPath.clear();
					maxPath.push_back(currentIterator);
					for(int i=0;i<path.size();i++)
						maxPath.push_back(path[i]);
				}
			}
			currentIterator++;
		}
		
		for(int i=0;i<maxPath.size();i++)
		{
			tree<NodeFunctionRetain>::iterator treeLeaf = maxPath[i];
			NodeFunctionRetain nodeFunction = *treeLeaf;
			nodeFunction.retainCount--;
			if(nodeFunction.retainCount == 0)
				nodes.erase(treeLeaf);
		}
	}
	
	void ImagePipeline::feed(cv::Mat image,void* context)
	{
		tree<NodeFunctionRetain>::sibling_iterator currentIterator = nodes.begin();
		tree<NodeFunctionRetain>::sibling_iterator endIterator = nodes.end();
		
		while(currentIterator != endIterator)
		{
			NodeFunctionRetain nodeFunction = *currentIterator;
			if(nodeFunction.type == &typeid(preprocessingFunc))
			{
				preprocessingFunc preprocessing = (preprocessingFunc) nodeFunction.function;
				ImagePipeline::depthTreeRun(currentIterator,preprocessing(image,context),context);
			}
			else if(nodeFunction.type == &typeid(analysisFunc))
			{
				analysisFunc analysis = (analysisFunc) nodeFunction.function;
				analysis(image,context);
			}
			else if(nodeFunction.type == &typeid(preprocessingSplitFunc))
			{
				preprocessingSplitFunc preprocessingSplit = (preprocessingSplitFunc) nodeFunction.function;
				std::vector<cv::Mat> images = preprocessingSplit(image,context);
				for(int i=0;i<images.size();i++)
					ImagePipeline::depthTreeRun(currentIterator,images[i],context);
			}
			currentIterator++;
		}
	}
	
	std::vector<tree<NodeFunctionRetain>::iterator> ImagePipeline::depthTreeRecursion(tree<NodeFunctionRetain>::iterator currentBranch,std::vector<NodeFunction>::iterator currentNode,std::vector<NodeFunction>::iterator endNode)
	{
		std::vector<tree<NodeFunctionRetain>::iterator> maxPath;
		
		if(currentNode == endNode)
			return maxPath;
		
		tree<NodeFunctionRetain>::sibling_iterator currentIterator = currentBranch.begin();
		tree<NodeFunctionRetain>::sibling_iterator endIterator = currentBranch.end();
		
		while(currentIterator != endIterator)
		{
			NodeFunction nodeFunction = *currentNode;
			NodeFunctionRetain treeFunction = *currentIterator;
			if(nodeFunction.function == treeFunction.function)
			{
				std::vector<tree<NodeFunctionRetain>::iterator> path = ImagePipeline::depthTreeRecursion(currentIterator,currentNode+1,endNode);
				if(path.size()+1 > maxPath.size())
				{
					maxPath.clear();
					maxPath.push_back(currentIterator);
					for(int i=0;i<path.size();i++)
						maxPath.push_back(path[i]);
				}
			}
			currentIterator++;
		}
		
		return maxPath;
	}
	
	void ImagePipeline::depthTreeRun(tree<NodeFunctionRetain>::iterator currentBranch,cv::Mat image,void* context)
	{
		tree<NodeFunctionRetain>::sibling_iterator currentIterator = currentBranch.begin();
		tree<NodeFunctionRetain>::sibling_iterator endIterator = currentBranch.end();
		
		while(currentIterator != endIterator)
		{
			NodeFunctionRetain nodeFunction = *currentIterator;
			if(nodeFunction.type == &typeid(preprocessingFunc))
			{
				preprocessingFunc preprocessing = (preprocessingFunc) nodeFunction.function;
				ImagePipeline::depthTreeRun(currentIterator,preprocessing(image,context),context);
			}
			else if(nodeFunction.type == &typeid(analysisFunc))
			{
				analysisFunc analysis = (analysisFunc) nodeFunction.function;
				analysis(image,context);
			}
			else if(nodeFunction.type == &typeid(preprocessingSplitFunc))
			{
				preprocessingSplitFunc preprocessingSplit = (preprocessingSplitFunc) nodeFunction.function;
				std::vector<cv::Mat> images = preprocessingSplit(image,context);
				for(int i=0;i<images.size();i++)
					ImagePipeline::depthTreeRun(currentIterator,images[i],context);
			}
			currentIterator++;
		}
	}
};
