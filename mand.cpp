//mand.cpp
#define SDL_MAIN_HANDLED
#include <vulkan/vulkan.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vector>
#include <math.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>

bool framebufferResized = false;
uint32_t windowWidth  = 1250;
uint32_t windowHeight = 670;
int initTime = 0.0;

VkDevice device;
VkPhysicalDevice physicalDevice;
VkSurfaceKHR surface;
VkSwapchainKHR swapchain;
VkSurfaceFormatKHR surfaceFormat;
VkExtent2D swapExtent;
std::vector<VkImage> images;
std::vector<VkImageView> imageViews;
std::vector<VkFramebuffer> framebuffers;
VkRenderPass renderPass;
VkPipeline pipeline;
VkPipelineLayout layout;

//std::string iterBuffer;
//int maxIterations;

struct PushConstants {
    			float width, height;
    			float time;
                float sinTime;
				float cosTime;
				float tanTime;
    			double zoom;
    			double centerX, centerY;
                int findex;
                int cindex;
				int maxIterations;
                //double zoomhi, zoomlo;
				//double centerXhi, centerXlo;
				//double centerYhi, centerYlo;
};

std::vector<char> readFile(const std::string &filename) {
    	std::ifstream file(filename, std::ios::ate | std::ios::binary);
    	if (!file) throw std::runtime_error("Failed to open file!");
    	size_t size = (size_t)file.tellg();
    	std::vector<char> buffer(size);
    	file.seekg(0);
    	file.read(buffer.data(), size);
    	return buffer;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code)
	{
    	VkShaderModuleCreateInfo createInfo{};
    	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    	createInfo.codeSize = code.size();
    	createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    	VkShaderModule shaderModule;
    	if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
        	throw std::runtime_error("Failed to create shader module!");
    	return shaderModule;
}

uint32_t findGraphicsQueueFamily(VkPhysicalDevice phys, VkSurfaceKHR surface) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, families.data());

    for (uint32_t i = 0; i < count; i++) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(phys, i, surface, &present);

        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present)
            return i;
    }
    throw std::runtime_error("No suitable queue family");
}

void recreateSwapchain()
{
    // Wait until window is non-zero sized
    while (windowWidth == 0 || windowHeight == 0) {
        SDL_Event e;
        SDL_WaitEvent(&e);
    }

    vkDeviceWaitIdle(device);

    // Destroy framebuffers
    for (auto fb : framebuffers)
        vkDestroyFramebuffer(device, fb, nullptr);
    framebuffers.clear();

    // Destroy image views
    for (auto iv : imageViews)
        vkDestroyImageView(device, iv, nullptr);
    imageViews.clear();

    // Destroy swapchain
    vkDestroySwapchainKHR(device, swapchain, nullptr);

    // Re-query surface capabilities
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &caps);

    swapExtent.width  = windowWidth;
    swapExtent.height = windowHeight;

    VkSwapchainCreateInfoKHR swapInfo{};
    swapInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapInfo.surface = surface;
    swapInfo.minImageCount = caps.minImageCount + 1;
    swapInfo.imageFormat = surfaceFormat.format;
    swapInfo.imageColorSpace = surfaceFormat.colorSpace;
    swapInfo.imageExtent = swapExtent;
    swapInfo.imageArrayLayers = 1;
    swapInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapInfo.preTransform = caps.currentTransform;
    swapInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    swapInfo.clipped = VK_TRUE;

    vkCreateSwapchainKHR(device, &swapInfo, nullptr, &swapchain);

    uint32_t imageCount;
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    images.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, images.data());

    imageViews.resize(imageCount);
    for (uint32_t i = 0; i < imageCount; i++) {
        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = images[i];
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = surfaceFormat.format;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.levelCount = 1;
        view.subresourceRange.layerCount = 1;
        vkCreateImageView(device, &view, nullptr, &imageViews[i]);
    }

    framebuffers.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++) {
        VkFramebufferCreateInfo fb{};
        fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb.renderPass = renderPass;
        fb.attachmentCount = 1;
        fb.pAttachments = &imageViews[i];
        fb.width  = swapExtent.width;
        fb.height = swapExtent.height;
        fb.layers = 1;
        vkCreateFramebuffer(device, &fb, nullptr, &framebuffers[i]);
    }
}

int main()
{
	VkDynamicState dynamics[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dyn{};
	dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dyn.dynamicStateCount = 2;
	dyn.pDynamicStates = dynamics;

	//std::ofstream log("trace.log", std::ios::app);
	//log << "main() start" << std::endl;
    	// --- Window ---
	if (SDL_Init(SDL_INIT_VIDEO) != 0){SDL_Log("SDL_Init failed: %s", SDL_GetError()); return -1;}
	SDL_Window* window = SDL_CreateWindow("VulkanTest", SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED, windowWidth, windowHeight, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN);
	if (!window){SDL_Log("SDL_CreateWindow failed: %s", SDL_GetError()); return -1;}

	double zoom = 1.5;
	double centerX = -0.75;
	double centerY =  0.0;
	//double centerXhi, centerXlo = 0.0;
	//double centerYhi, centerYlo = 0.0;
	//double zoomhi, zoomlo = 0.0;
    //long double zoom_acc = (long double)zoom;
    //long double centerX_acc = (long double)centerX;
    //long double centerY_acc = (long double)centerY;

    // --- Vulkan Instance ---
    VkInstance instance;
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "MinimalShader";
    appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
    appInfo.apiVersion = VK_API_VERSION_1_0;
//log << "1" << std::endl;
unsigned int extCount = 0;
if (!SDL_Vulkan_GetInstanceExtensions(window, &extCount, nullptr))
    throw std::runtime_error("Failed to get SDL Vulkan extensions count");

std::vector<const char*> extensions(extCount);
if (!SDL_Vulkan_GetInstanceExtensions(window, &extCount, extensions.data()))
    throw std::runtime_error("Failed to get SDL Vulkan extensions list");

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

createInfo.enabledExtensionCount = extCount;
createInfo.ppEnabledExtensionNames = extensions.data();

//log << "2" << std::endl;
    	if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) throw std::runtime_error("Failed to create Vulkan instance!");
//log << "3" << std::endl;
	if (!SDL_Vulkan_CreateSurface(window, instance, &surface)) throw std::runtime_error("Failed to create Vulkan surface");
//log << "5" << std::endl;
	uint32_t physicalCount = 0;
//log << "6" << std::endl;
	vkEnumeratePhysicalDevices(instance, &physicalCount, nullptr);
	if (physicalCount == 0) throw std::runtime_error("No Vulkan devices");
//log << "7" << std::endl;
	std::vector<VkPhysicalDevice> physicalDevices(physicalCount);
	vkEnumeratePhysicalDevices(instance, &physicalCount, physicalDevices.data());
//log << "6" << std::endl;
	physicalDevice = physicalDevices[0];
//log << "3" << std::endl;
	int32_t queueFamily = findGraphicsQueueFamily(physicalDevice, surface);
	float priority = 1.0f;
	VkDeviceQueueCreateInfo queueInfo{};
	queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueInfo.queueFamilyIndex = queueFamily;
	queueInfo.queueCount = 1;
	queueInfo.pQueuePriorities = &priority;
//log << "4" << std::endl;
	VkDeviceCreateInfo deviceInfo{};
	deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceInfo.queueCreateInfoCount = 1;
	deviceInfo.pQueueCreateInfos = &queueInfo;
//log << "5" << std::endl;
	const char* deviceExtensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
	deviceInfo.enabledExtensionCount = 1;
	deviceInfo.ppEnabledExtensionNames = deviceExtensions;
//log << "6" << std::endl;
	if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device) != VK_SUCCESS)
    	throw std::runtime_error("Failed to create logical device");

VkSurfaceCapabilitiesKHR caps;
vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &caps);

uint32_t formatCount = 0;
vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
std::vector<VkSurfaceFormatKHR> formats(formatCount);
vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data());
surfaceFormat = formats[0];
//log << "mid1" << std::endl;
VkSwapchainCreateInfoKHR swapInfo{};
swapInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
swapInfo.surface = surface;
swapInfo.minImageCount = caps.minImageCount + 1;
swapInfo.imageFormat = surfaceFormat.format;
swapInfo.imageColorSpace = surfaceFormat.colorSpace;
swapInfo.imageExtent = caps.currentExtent;
swapInfo.imageArrayLayers = 1;
swapInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
swapInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
swapInfo.preTransform = caps.currentTransform;
swapInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
swapInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
swapInfo.clipped = VK_TRUE;
vkCreateSwapchainKHR(device, &swapInfo, nullptr, &swapchain);
swapExtent = caps.currentExtent;

uint32_t imageCount = 0;
vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
std::vector<VkImage> images(imageCount);
vkGetSwapchainImagesKHR(device, swapchain, &imageCount, images.data());
//log << "mid2" << std::endl;
std::vector<VkImageView> imageViews(imageCount);
for (uint32_t i = 0; i < imageCount; i++) {
    VkImageViewCreateInfo view{};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = images[i];
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = surfaceFormat.format;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.levelCount = 1;
    view.subresourceRange.layerCount = 1;
    vkCreateImageView(device, &view, nullptr, &imageViews[i]);
}

VkAttachmentDescription color{};
color.format = surfaceFormat.format;
color.samples = VK_SAMPLE_COUNT_1_BIT;
color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
//log << "mid3" << std::endl;
VkAttachmentReference colorRef{};
colorRef.attachment = 0;
colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

VkSubpassDescription subpass{};
subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
subpass.colorAttachmentCount = 1;
subpass.pColorAttachments = &colorRef;

VkRenderPassCreateInfo rp{};
rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
rp.attachmentCount = 1;
rp.pAttachments = &color;
rp.subpassCount = 1;
rp.pSubpasses = &subpass;

vkCreateRenderPass(device, &rp, nullptr, &renderPass);
//log << "mid4" << std::endl;
std::vector<VkFramebuffer> framebuffers(imageCount);
for (size_t i = 0; i < imageCount; i++) {
    VkFramebufferCreateInfo fb{};
    fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb.renderPass = renderPass;
    fb.attachmentCount = 1;
    fb.pAttachments = &imageViews[i];
    fb.width = caps.currentExtent.width;
    fb.height = caps.currentExtent.height;
    fb.layers = 1;
    vkCreateFramebuffer(device, &fb, nullptr, &framebuffers[i]);
}


	VkQueue graphicsQueue;
	vkGetDeviceQueue(device, queueFamily, 0, &graphicsQueue);


    	auto vertCode = readFile("fullscreen.vert.spv");
	auto fragCode = readFile("testShader.frag.spv");

	VkShaderModule vert = createShaderModule(device, vertCode);
	VkShaderModule frag = createShaderModule(device, fragCode);


	VkPipelineShaderStageCreateInfo stages[2]{};

	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vert;
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = frag;
	stages[1].pName = "main";

VkPipelineVertexInputStateCreateInfo vi{};
vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

VkPipelineInputAssemblyStateCreateInfo ia{};
ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

VkViewport viewport{};
viewport.width = (float)caps.currentExtent.width;
viewport.height = (float)caps.currentExtent.height;
viewport.maxDepth = 1.0f;

VkRect2D scissor{};
scissor.extent = caps.currentExtent;

VkPipelineViewportStateCreateInfo vp{};
vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
vp.viewportCount = 1;
vp.pViewports = &viewport;
vp.scissorCount = 1;
vp.pScissors = &scissor;

VkPipelineRasterizationStateCreateInfo rs{};
rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
rs.polygonMode = VK_POLYGON_MODE_FILL;
rs.lineWidth = 1.0f;
rs.cullMode = VK_CULL_MODE_NONE;
rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
//log << "mid5" << std::endl;
VkPipelineMultisampleStateCreateInfo ms{};
ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

VkPipelineColorBlendAttachmentState cb{};
cb.colorWriteMask = 0xF;

VkPipelineColorBlendStateCreateInfo cbState{};
cbState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
cbState.attachmentCount = 1;
cbState.pAttachments = &cb;

VkPushConstantRange pushRange{};
pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
pushRange.offset = 0;
pushRange.size = sizeof(PushConstants);
VkPipelineLayoutCreateInfo layoutInfo{};
layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
layoutInfo.pushConstantRangeCount = 1;
layoutInfo.pPushConstantRanges = &pushRange;
if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create pipeline layout");

VkGraphicsPipelineCreateInfo pipe{};
pipe.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
pipe.stageCount = 2;
pipe.pStages = stages;
pipe.pVertexInputState = &vi;
pipe.pInputAssemblyState = &ia;
pipe.pViewportState = &vp;
pipe.pRasterizationState = &rs;
pipe.pMultisampleState = &ms;
pipe.pColorBlendState = &cbState;
pipe.layout = layout;
pipe.renderPass = renderPass;
pipe.pDynamicState = &dyn;
vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipe, nullptr, &pipeline);

VkCommandPool pool;
VkCommandPoolCreateInfo poolInfo{};
poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
poolInfo.queueFamilyIndex = queueFamily;
vkCreateCommandPool(device, &poolInfo, nullptr, &pool);

VkCommandBuffer cmd;
VkCommandBufferAllocateInfo alloc{};
alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
alloc.commandPool = pool;
alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
alloc.commandBufferCount = 1;
vkAllocateCommandBuffers(device, &alloc, &cmd);

const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
createInfo.enabledLayerCount = 1;
createInfo.ppEnabledLayerNames = layers;

	//log << "end init" << std::endl;
	uint64_t perfFreq = SDL_GetPerformanceFrequency();
	uint64_t lastCounter = SDL_GetPerformanceCounter();
	double fps = 0.0;
	int frames = 0;
	double fpsTimer = 0.0;
	int frameNum = 0;
    int findex = 0;
    int cindex = 0;
	int maxIterations = 256;
	bool running = true;
	while (running)
	{
		uint32_t imageIndex;
		vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, VK_NULL_HANDLE, VK_NULL_HANDLE, &imageIndex);

		vkResetCommandBuffer(cmd, 0);
		VkCommandBufferBeginInfo begin{};
		begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		vkBeginCommandBuffer(cmd, &begin);

		VkClearValue clear{{0.0f, 0.0f, 0.0f, 1.0f}};
		VkRenderPassBeginInfo rpBegin{};
		rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		rpBegin.renderPass = renderPass;
		rpBegin.framebuffer = framebuffers[imageIndex];
		rpBegin.renderArea.extent = caps.currentExtent;
		rpBegin.clearValueCount = 1;
		rpBegin.pClearValues = &clear;

		PushConstants pc{};
		pc.width  = (float)caps.currentExtent.width;
		pc.height = (float)caps.currentExtent.height;
		uint64_t counter = SDL_GetPerformanceCounter();
        if (!initTime){
            pc.time = 0.0f;
            initTime = (float)counter / (float)perfFreq;
        }
        else {
			pc.time += ((float)counter / (float)perfFreq) - (float)initTime;
        }
		pc.sinTime = sinf(pc.time * 0.5f);
		pc.cosTime = cosf(pc.time * 0.5f);
		pc.tanTime = tanf(pc.time * 0.5f);
		pc.zoom = zoom;
		pc.centerX = centerX;
		pc.centerY = centerY;
		pc.findex = findex;
		pc.cindex = cindex;
		pc.maxIterations = maxIterations;
        /*pc.zoomhi = (double)zoom_acc;
        pc.zoomlo = (double)(zoom_acc - (long double)pc.zoomhi);
        pc.centerXhi = (double)centerX_acc;
        pc.centerXlo = (double)(centerX_acc - (long double)pc.centerXhi);
        pc.centerYhi = (double)centerY_acc;
        pc.centerYlo = (double)(centerY_acc - (long double)pc.centerYhi);*/

		VkViewport vp{};
		vp.width  = (float)swapExtent.width;
		vp.height = (float)swapExtent.height;
		vp.maxDepth = 1.0f;
		VkRect2D sc{};
		sc.extent = swapExtent;

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		vkCmdSetViewport(cmd, 0, 1, &vp);
		vkCmdSetScissor(cmd, 0, 1, &sc);
		vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);
		vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdDraw(cmd, 3, 1, 0, 0);

		vkCmdEndRenderPass(cmd);
		vkEndCommandBuffer(cmd);

		// Submit the command buffer
    		VkSubmitInfo submit{};
    		submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    		submit.commandBufferCount = 1;
    		submit.pCommandBuffers = &cmd;
    		vkQueueSubmit(graphicsQueue, 1, &submit, VK_NULL_HANDLE);
    		vkQueueWaitIdle(graphicsQueue);  // minimal; for real apps you want fences

    		// Present
    		VkPresentInfoKHR present{};
    		present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    		present.swapchainCount = 1;
    		present.pSwapchains = &swapchain;
    		present.pImageIndices = &imageIndex;
    		vkQueuePresentKHR(graphicsQueue, &present);
		//if(!frameNum) log << "end of loop reached, loop number:" << std::endl;
		//log << frameNum << std::endl;
		frameNum++;
    		SDL_Event e;
    		while (SDL_PollEvent(&e))
		{
			switch(e.type){
        			case SDL_QUIT: running = false; break;
				case SDL_WINDOWEVENT:{
    					if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED){
					windowWidth  = e.window.data1;
        				windowHeight = e.window.data2;
        				framebufferResized = true;
    					}
    					break;
				}
                case SDL_MOUSEWHEEL:
                {
                    int mx, my; // Get mouse position in window pixels
                    SDL_GetMouseState(&mx, &my);

                    // Use plain double math (drop long-double accumulators)
                    double prevZoom = zoom;
                    double zoomFactor = (e.wheel.y > 0) ? 0.9 : 1.1; // Zoom factor
                    zoom *= zoomFactor;

                    // normalized coordinates (-1..1) with aspect correction
                    double nx = (2.0 * mx / (double)windowWidth - 1.0);
                    double ny = (2.0 * my / (double)windowHeight - 1.0);
                    double aspect = (double)windowWidth / (double)windowHeight;
                    nx *= aspect;

                    // world-space delta applied in double precision
                    double scale = prevZoom - zoom;
                    centerX += nx * scale;
                    centerY += ny * scale;
                    break;
                }
                case SDL_MOUSEBUTTONDOWN:
                {
                        int mx = e.button.x;
                        int my = e.button.y;
                        double prevZoom = zoom;
                        double zoomFactor = 1.0;
                        if (e.button.button == SDL_BUTTON_LEFT) zoomFactor = 0.9;   // zoom in
                        else if (e.button.button == SDL_BUTTON_RIGHT) zoomFactor = 1.1; // zoom out
                        zoom *= zoomFactor;
                        double nx = (2.0 * mx / (double)windowWidth - 1.0);
                        double ny = (2.0 * my / (double)windowHeight - 1.0);
                        double aspect = (double)windowWidth / (double)windowHeight;
                        nx *= aspect;
                        double scale = prevZoom - zoom;
                        centerX += nx * scale;
                        centerY += ny * scale;
                        break;
                }
                case SDL_KEYDOWN: {
                    SDL_Keycode k = e.key.keysym.sym;
                    if (k == SDLK_F11) {
                        Uint32 flags = SDL_GetWindowFlags(window);
                        bool fullscreen = flags & SDL_WINDOW_FULLSCREEN_DESKTOP;
                        SDL_SetWindowFullscreen(window, fullscreen ? 0 : SDL_WINDOW_FULLSCREEN_DESKTOP);
                    }
                    else if (k == SDLK_LEFT) {
                        findex = findex - 1;
                    }
                    else if (k == SDLK_RIGHT) {
                        findex = findex + 1;
                    }
                    else if (k == SDLK_UP) {
                        cindex = cindex + 1;
                    }
                    else if (k == SDLK_DOWN) {
                        cindex = cindex - 1;
                    }
                    else if (k == SDLK_MINUS || k == SDLK_KP_MINUS) {
                        maxIterations = maxIterations - 16; // step down
                    }
                    else if (k == SDLK_EQUALS || k == SDLK_KP_PLUS) {
                        maxIterations = maxIterations + 16; // step up
                    }
                    break;
                }
    			}
		}
		if (framebufferResized) {
    			vkDeviceWaitIdle(device);
    			recreateSwapchain();
			vkDestroyPipeline(device, pipeline, nullptr);
			//createPipeline(); // must rebind renderPass, viewport state, layout

    			framebufferResized = false;
		}
		uint64_t now = SDL_GetPerformanceCounter();
		double deltaTime = (double)(now - lastCounter) / (double)perfFreq;
		lastCounter = now;
		fpsTimer += deltaTime;
		frames++;
		if (fpsTimer >= 0.5)
		{
    			fps = frames / fpsTimer;
    			frames = 0;
    			fpsTimer = 0.0;
			char title[128];
			snprintf(title, sizeof(title), "Test — FPS = %.1f, Index = %.1u, Pallete = %.1u, MaxI = %.1u Time = %.1f Zoom = %.9g, x = %.9g, y = %.9g", fps, pc.findex, pc.cindex, pc.maxIterations, pc.time, (double)pc.zoom,  (double)pc.centerX, (double)pc.centerY);
			SDL_SetWindowTitle(window, title);
		}
	}
	vkDestroyDevice(device, nullptr);
	vkDestroySurfaceKHR(instance, surface, nullptr);
    	vkDestroyInstance(instance, nullptr);
    	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
