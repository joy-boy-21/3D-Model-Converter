import matplotlib
from matplotlib import pyplot as plt 
from PIL import Image
import torch
#from transformers import GLPNImageProcessor,GLPNForDepthEstimation

# Use a pipeline as a high-level helper
from transformers import pipeline

# Load model directly
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

featureExtracter = AutoImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = AutoModelForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

pipe = pipeline("depth-estimation", model="vinvino02/glpn-nyu")

#featureExtracter=GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
#model=GLPNForDepthEstimation.from_pretrained("vinvino02.glpn-nyu")

image=Image.open(r"C:\Users\S.Sanjaikumar\CODE\MiniProjects\3D MODEL\data\birdd.jpg")

newHeight=480 if image.height>480 else image.height
newHeight-=(newHeight%32)
newWidth=int(newHeight*image.width/image.height)
dif=newWidth%32

newWidth=newWidth-dif if dif<16 else newHeight+32-dif
newSize=(newWidth,newHeight)
image=image.resize(newSize)

inputs=featureExtracter(images=image,return_tensors="pt")

with torch.no_grad():
    outputs=model(**inputs)
    predicted_depth=outputs.predicted_depth

pad=16
output=predicted_depth.squeeze().cpu().numpy()*1000.0
output=output[pad:-pad,pad:-pad]
image=image.crop((pad,pad,image.width -pad,image.height - pad))

"""fig,ax=plt.subplots(1,2)
ax[0].imshow(image)
ax[1].tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
ax[1].imshow(output,cmap='plasma')
ax[1].tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
plt.tight_layout()
plt.pause(5)"""

import numpy as np
import open3d as o3d 

width,height=image.size

depth_image=(output*255/np.max(output)).astype('uint8')
image=np.array(image)

depth_o3d=o3d.geometry.Image(depth_image)
image_o3d = o3d.geometry.Image(image)
rgbd_image=o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d,depth_o3d,convert_rgb_to_intensity=False)

cam_intrinsic=o3d.camera.PinholeCameraIntrinsic()
cam_intrinsic.set_intrinsics(width,height,500,500,width/2,height/2)

pcdRaw=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,cam_intrinsic)

#o3d.visualization.draw_geometries([pcdRaw])


cl,ind=pcdRaw.remove_statistical_outlier(nb_neighbors=20,std_ratio=20.0)
pcd=pcdRaw.select_by_index(ind)

pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction()

o3d.visualization.draw_geometries([pcd])




