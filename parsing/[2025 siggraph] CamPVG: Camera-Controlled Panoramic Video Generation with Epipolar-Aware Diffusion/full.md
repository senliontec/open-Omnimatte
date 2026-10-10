# CamPVG: Camera-Controlled Panoramic Video Generation with Epipolar-Aware Diffusion

CHENHAO JI\*, Tongji University, China and DAMO Academy, Alibaba group, China

CHAOHUI YU, DAMO Academy, Alibaba group, China

JUNYAO GAO, Tongji University, China

FAN WANG, DAMO Academy, Alibaba group, USA

CAIRONG ZHAO†, Tongji University, China

![](images/724e596efd193500de6a9b8631cbb041ae1c6da55cc2d9c4ad84b5edba3f793b.jpg)  
Fig. 1. CamPVG is the first camera-controlled panoramic video generation framework. Given a specified camera trajectory and an initial conditional frame, it generates high-quality panoramic videos with strong geometric consistency across the panoramic space. By leveraging panoramic Plücker embeddings and a spherical epipolar-aware module, our method effectively models global geometric structures and viewpoint transitions, generating spatially coherent and visually realistic panoramic videos.

\*Work done during an internship in DAMO Academy, Alibaba Group.   
†Corresponding author.

Authors' Contact Information: Chenhao Ji, Tongji University, Shanghai, China and DAMO Academy, Alibaba group, Shanghai, China, jichenhao@tongji.edu.cn; Chaohui Yu, DAMO Academy, Alibaba group, Beijing, China, huakun.ych@alibaba-inc.com; Junyao Gao, Tongji University, Shanghai, China, junyaogao@tongji.edu.cn; Fan Wang, DAMO Academy, Alibaba group, SunnyVale, USA, fan.w@alibaba-inc.com; Cairong Zhao, Tongji University, Shanghai, China, zhaocairong@tongji.edu.cn.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed

for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.   
SA Conference Papers '25, Hong Kong, Hong Kong   
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-2137-3/25/12   
https://doi.org/10.1145/3757377.3763990

Recently, camera-controlled video generation has seen rapid development, offering more precise control over video generation. However, existing methods predominantly focus on camera control in perspective projection video generation, while geometrically consistent panoramic video generation remains challenging. This limitation is primarily due to the inherent complexities in panoramic pose representation and spherical projection. To address this issue, we propose CamPVG, the first diffusion-based framework for panoramic video generation guided by precise camera poses. We achieve camera position encoding for panoramic images and cross-view feature aggregation based on spherical projection. Specifically, we propose a panoramic Plücker embedding that encodes camera extrinsic parameters through spherical coordinate transformation. This pose encoder effectively captures panoramic geometry, overcoming the limitations of traditional methods when applied to equirectangular projections. Additionally, we introduce a spherical epipolar module that enforces geometric constraints through adaptive attention masking along epipolar lines. This module enables fine-grained cross-view feature aggregation, substantially enhancing the quality and consistency of generated panoramic videos. Extensive experiments demonstrate that our method generates high-quality panoramic videos consistent with camera trajectories, far surpassing existing methods in panoramic video generation.

CCS Concepts: • Computing methodologies → Computer vision.

Additional Key Words and Phrases: AIGC, panoramic video generation, camera pose guidance, spherical epipolar geometry, video diffusion models

## ACM Reference Format:

Chenhao Ji, Chaohui Yu, Junyao Gao, Fan Wang, and Cairong Zhao. 2025. CamPVG: Camera-Controlled Panoramic Video Generation with Epipolar-Aware Diffusion. In SIGGRAPH Asia 2025 Conference Papers (SA Conference Papers '25), December 15–18, 2025, Hong Kong, Hong Kong. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3757377.3763990

## 1 Introduction

The rapid advancement of virtual reality (VR), metaverse technologies, and embodied artificial intelligence has catalyzed a surge of interest in panoramic visual content. Panoramic videos, which capture a comprehensive 360-degree view of the surrounding environment, offer users an immersive experience. As the applications of panoramic videos expand, the demand for enhanced immersive experiences has continually increased. The development of scalable panoramic video generation with precise camera pose control has emerged as a critical area of research. This capability opens new frontiers for expansive applications in entertainment, interaction, and beyond.

Generating panoramic videos with consistent camera motion and temporal smoothness poses distinctive challenges. Many existing works on panoramic video generation, such as 360DVD [Wang et al. 2024a], Imagine 360 [Tan et al. 2024], and 4K4DGen [Li et al. 2024], primarily focus on generating dynamic content in panoramic videos but offer limited control over camera perspectives. Consequently, the generated panoramic videos exhibit minimal variation in viewpoints. To achieve precise camera pose control, recent approaches have made notable strides. For instance, MotionCtrl [Wang et al. 2024b] concatenates camera poses with features in the latent space, while CameraCtrl [He et al. 2024] injects camera poses into the latent space using Plücker embedding. In addition, CamCo [Xu et al. 2024] and CamI2V [Zheng et al. 2024] further enhance scene consistency across different camera viewpoints by introducing geometric constraints through epipolar attention. However, these methods are specifically designed for perspective projection video generation and show limitations when applied to panoramic domain. These limitations stem primarily from the challenging representation of panoramic camera poses and the intrinsic geometric complexity of panoramic imagery.

To address these challenges, we propose CamPVG, the first diffusion· based framework for panoramic video generation guided by precise camera poses. Our approach enables the generation of high-quality panoramic videos that maintain consistency with given camera trajectories as shown in Fig. 1. Unlike prior perspective-based geometric methods, our approach is not simply an adaptation of perspective models to panoramic coordinate systems. Existing camera pose encoding methods are typically designed for perspective projection camera trajectories and perform poorly on equirectangular panoramic data due to the inherent differences in imaging logic between panoramic and perspective views. To achieve effective camera position encoding for panoramic data, we introduce panoramic Plücker embedding, building upon the foundation of traditional Plücker embedding [He et al. 2024]. Our method models the spatial relationship between each pixel in the panoramic image and the camera origin through spherical projection, representing this relationship using Plücker coordinates [Sitzmann et al. 2021]. These spherically projected Plücker coordinates are then injected into the latent space via a pose encoder, providing spatial geometric guidance throughout the panoramic video generation process. Furthermore, to enhance consistency across different camera viewpoints, we propose a spherical epipolar module for fine-grained feature aggregation. By leveraging the intrinsic properties of equirectangular projection, we calculate the spherical epipolar lines corresponding to each pixel across different viewpoints. We then employ spherical epipolar masking with carefully designed sampling strategy along the epipolar line to filter out irrelevant pixel information. During the panoramic video generation process, we aggregate valid reference information from different viewpoints using spherical epipolar attention, thereby achieving multi-view consistent panoramic video generation.

Extensive experiments demonstrate that CamPVG achieves superior performance in camera trajectory consistency, frame realism, and overall video quality. Our method significantly surpasses existing camera-controlled video generation approaches in panoramic video generation. We believe that CamPVG will make substantial contributions to the field of camera pose-guided panoramic video generation and its downstream applications. Our contributions can be summarized as follows:

• We propose CamPVG, the first framework for panoramic video generation guided by precise camera poses, enabling the generation of high-quality panoramic videos with consistent camera trajectories.

• We introduce panoramic Plücker embedding, a novel approach for camera position encoding based on panoramic data.

• We present the spherical epipolar module that leverages spherical epipolar constraints to achieve fine-grained feature aggregation, enhancing multi-view consistency and visual fidelity of panoramic videos.

## 2 Related Work

## 2.1 Diffusion-Based Video Generation

Recent advancements in diffusion models [Gao et al. 2025a, 2024, 2025b; Jiang et al. 2024b,a; Tang et al. 2025] have significantly advanced research in video generation. Building on the success of text-to-image (T2I) diffusion frameworks such as Stable Diffusion [Rombach et al. 2022], researchers have extended these models to text-to-video (T2V) generation by incorporating temporal layers to process video input while retaining strong visual priors. For instance, Video Diffusion Model [Ho et al. 2022], LVDM [He et al. 2022], and VideoCrafter [Chen et al. 2023, 2024] extend the 2D U-Net architecture of image diffusion models with spatial and temporal blocks, enabling coherent video generation through iterative denoising. Furthermore, Sora [Brooks et al. 2024] and CogVideoX [Yang et al. 2024] explore Transfomer-based diffusion framework integrated with 3D-VAE, significantly enhanced the video generation capabilities in terms of temporal consistency and visual fidelity. Additionally, other works [Blattmann et al. 2023; Xing et al. 2024] have advanced image-to-video (I2V) generation by conditioning diffusion models on image inputs.

## 2.2 Panoramic Video Generation

The field of panoramic content generation has seen notable progress with the advent of diffusion models, though most efforts remain focused on static panorama synthesis [Koh et al. 2021; Tang et al. 2023; Ye et al. 2024; Yuan et al. 2025; Zhang et al. 2024] rather than video generation. Recent studies [Liu et al. 2025; Xie et al. 2025] have increasingly explored diffusion-based frameworks to overcome these limitations. For instance, 360DVD [Wang et al. 2024a] introduces a lightweight 360-Adapter to fine-tune pre-trained T2I diffusion models, enabling panoramic video synthesis conditioned on textual prompts and motion signals. Imagine360 [Tan et al. 2024] proposes a dual-branch architecture that enforces joint local and global constraints, facilitating perspective-to-panoramic video conversion. Another approach, 4K4DGen [Li et al. 2024], leverages 2D priors from perspective image generation models to denoise spherical latent codes, yet the generated videos suffer from restricted viewpoint diversity due to inadequate motion modeling in the latent space. While these methods demonstrate promising progress, their controllability over camera trajectories remains limited. Our CamPVG advances the field by integrating explicit camera pose conditioning into the diffusion framework, enabling precise control over viewpoint transitions in panoramic video generation.

## 2.3 Camera-Controlled Video Generation

Camera-controlled video generation has emerged as a critical research direction in diffusion-based video generation, aiming to produce dynamic visual content aligned with predefined camera trajectories. Some approaches [Hu et al. 2024; Jain et al. 2024] achieve coarse camera motion control through training-free methods. To enable precise camera control, recent works integrate camera pose information into diffusion frameworks. MotionCtrl [Wang et al. 2024b] concatenates noisy latent features with camera pose in temporal blocks, allowing camera-conditioned generation. Similarly, CameraCtrl [He et al. 2024] encodes Plücker embeddings [Sitzmann et al. 2021] and injects them into the U-Net architecture. While these methods demonstrate improved controllability, their ability to model 3D spatial relationships still limits. Addressing this limitation, some methods [Kuang et al. 2024; Xu et al. 2024; Zheng et al. 2024] introduce epipolar attention mechanisms to explicitly model 3D geometric constraints. While effective for perspective-view generation, their effectiveness remains constrained in panoramic video generation. Our work extends this principle to panoramic domains by reformulating epipolar attention for equirectangular projections.

## 3 Method

In this section, we introduce our novel method for panoramic video generation guided by precise camera poses with spherical epipolar constraints, as illustrated in Fig. 2. We begin with the preliminary concepts of controllable video diffusion models and the representation of camera poses in Sec. 3.1. To encode camera trajectories for panoramas, we propose a panoramic Plücker embedding in Sec. 3.2. To better capture geometric constraints between multi-view panoramic frames, Sec. 3.3 details the proposed spherical epipolar module.

## 3.1 Preliminary

3.1.1 Controllable Video Diffusion Model. Modern diffusion-based video generation frameworks synthesize content guided by multimodal conditional inputs. These models enable user-specified video synthesis by conditioning the generation process on diverse signals, including textual prompts, reference image, and motion information. The framework operates in a compressed latent space derived through a learned auto-encoder architecture. Given an input video sequence x $\in \mathbb { R } ^ { N \times H \times W \times 3 }$ comprising N frames of resolution $H { \times } W$ the encoder & produces latent representations $z _ { 0 } ^ { 1 : N } = \mathcal { E }$ . During training, Gaussian noise $\epsilon \sim { \cal N } ( 0 , \mathbf { I } )$ is progressively added across t diffusion steps, producing the noised latent $z _ { t } ^ { 1 : N }$ . The denoising model $\epsilon _ { \theta }$ learns to predict the noise e conditioned on the input signals at the time step t. The training objective can be formulated as follows:

$$
\mathcal { L } = \mathbb { E } _ { \mathcal { E } ( x ) , \epsilon \sim \mathcal { N } ( 0 , \mathrm { I } ) , c _ { t } , t } \left[ \left\| \epsilon - \hat { \epsilon } _ { \theta } \left( z _ { t } ^ { 1 : N } , c _ { t } , t \right) \right\| _ { 2 } ^ { 2 } \right] ,\tag{1}
$$

where $c _ { t }$ represents the embeddings of conditional information. This formulation enables joint optimization of spatial-temporal coherence and conditional alignment across modalities.

3.1.2 Camera Representation. The pose of the camera is defined by both intrinsic and extrinsic parameters. The intrinsic $\mathbf { K } \in \mathbb { R } ^ { 3 \times 3 }$ establishes the mapping from the camera coordinate system to the pixel coordinate system. The extrinsic $\mathbf { E } \in \mathbb { R } ^ { 3 \times 4 }$ , which includes a rotation matrix $\mathbf { R } \in S \mathbf { O } ( 3 )$ and a translation vector $\mathbf { t } \in \mathbb { R } ^ { 3 }$ , specifies the camera's orientation and position in the world coordinate system. Alternatively, camera poses can be encoded through Plücker embeddings [Sitzmann et al. 2021], which parameterize the relationship between image pixels and 3D rays originating from the camera. For each pixel (u, v), its Plücker embedding $\mathbf { P } _ { u , v } = \left( \mathbf { m } , \mathbf { d } \right) \in \mathbb { R } ^ { 6 }$ is defined as follows:

$\mathbf { d } \in \mathbb { R } ^ { 3 }$ represents the direction of the 3D ray from the camera center to the pixel in world coordinates.

![](images/ff1ddfc30c65f3366b5a420fb3426f44b2ac7eae843b6329417b67a75cfda197.jpg)  
Fig. 2. Framework of CamPVG. CamPVG employs spherical projection to transform input camera trajectories into panoramic Plücker embeddings, which are injected into the U-Net to guide panoramic geometry learning. Additionally, the spherical epipolar module computes epipolar masks through cross-view geometric constraints and applies spherical epipolar attention to enhance multi-view consistency. This integrated approach enables precise camera-controlled panoramic video generation.

• m $\epsilon \mathbb { R } ^ { 3 }$ denotes the moment vector, computed as the cross product between the position of the camera center and the direction vector d.

Given camera-to-world extrinsic $\mathbf { E } = \left[ \mathbf { R } , \mathbf { t } \right]$ and intrinsic K, the Plücker embedding for pixel $( u , v )$ is derived via:

$$
\mathbf { d } = \mathbf { R } \left( \mathbf { K } ^ { - 1 } \left( u , v , 1 \right) ^ { \top } \right) , \quad \mathbf { m } = \mathbf { t } \times \mathbf { d } .\tag{2}
$$

## 3.2 Panoramic Camera Pose Representation

Directly incorporating camera extrinsic parameters into video diffusion for learning viewpoint transitions poses significant challenges. To address this, we leverage Plücker embeddings that explicitly model the geometric relationship between pixels and camera rays. Compared with raw extrinsics, Plücker embeddings offer better numerical stability and richer geometric cues through their uniform magnitude distribution across the scene. However, conventional Plücker computation assumes perspective projection with known intrinsic parameters, which are undefined for equirectangular panoramas. For panoramas represented in equirectangular projection, we derive a spherical direction vector for each pixel via spherical projection and then calculate the corresponding panoramic plunker embedding, as illustrated in Fig. 3 (Left). Given a pixel $( u , v )$ in a panorama with resolution $H \times W$ , its corresponding spherical coordinates are computed as:

$$
\phi = \frac { u } { W } \cdot 2 \pi , \quad \theta = \frac { v } { H } \cdot \pi .\tag{3}
$$

Through the calculated azimuth φ and elevation θ angles in spherical coordinates, we can transform these into directional vectors in the Cartesian coordinate system:

$$
\begin{array} { r } { x _ { ( u , v ) } = \cos ( \theta ) \cdot \sin ( \phi ) , y _ { ( u , v ) } = \sin ( \theta ) , z _ { ( u , v ) } = \cos ( \theta ) \cdot \cos ( \phi ) . } \end{array}\tag{4}
$$

This spherical-to-cartesian conversion establishes a consistent 3D position mapping for panorama pixels, enabling Plücker embedding computation without conventional camera intrinsics. According to $\operatorname { E q . } ( 2 )$ , the Plücker embeddings for each panorama pixel with extrinsic $\mathbf { E } = \left[ \mathbf { R } , \mathbf { t } \right]$ are computed as:

$$
\mathbf { d } = \mathbf { R } \left( \hat { x } _ { ( u , v ) } , \hat { y } _ { ( u , v ) } , \hat { z } _ { ( u , v ) } \right) ^ { \top } , \quad \mathbf { m } = \mathbf { t } \times \mathbf { d } ,\tag{5}
$$

where $\left( \hat { x } _ { ( u , v ) } , \hat { y } _ { ( u , v ) } , \hat { z } _ { ( u , v ) } \right) ^ { \top }$ represents the normalized direction vector. We construct the complete camera trajectory $\mathbf { P } \in \mathbb { R } ^ { N \times H \times W \times 6 }$ by converting each panoramic video frame's camera extrinsics into Plücker embeddings. Following CameraCtrl [He et al. 2024], we employ a trainable pose encoder with linear projection layer to map the trajectory into latent representations. These encoded camera features are subsequently integrated into the diffusion U-Net to enable camera-aware generation.

## 3.3 Spherical Epipolar Module

3.3.1 Spherical Epipolar Line. Epipolar geometry establishes geometric constraints for potential pixel correspondences across multiview images. In perspective projection, epipolar lines can be directly computed through the essential matrix derived from relative camera poses and intrinsic parameters, resulting in straight lines in planar images. For equirectangular panoramas, however, the equirectangular projection necessitates a modified approach to epipolar geometry due to the non-linear coordinate mapping, as shown in Fig. 3 (Right). Given two panoramic views with extrinsics [R, t] and $\left[ \mathbf { R } _ { j } , \mathbf { t } _ { j } \right]$ , we compute their relative pose as:

$$
\mathbf { R } _ { i  j } = \mathbf { R } _ { j } \cdot \mathbf { R } _ { i } ^ { - 1 } , \quad \mathbf { t } _ { i  j } = \mathbf { t } _ { j } - \mathbf { R } _ { i  j } \mathbf { t } _ { i } .\tag{6}
$$

![](images/47528436a19c0c72209c0d2a2f7b58c33e89d72d265c6382987eb5daa56a7082.jpg)

![](images/0f88b46167a638d9fb8e80fa9bff828f5b785c05470e7ab24a8add0e919f857e.jpg)  
Fig. 3. Panoramic Plücker Embedding and Epipolar Geometry. Left: transformation from pixel coordinates to Panoramic Plücker embedding. Right: epipolar geometry relationship of pixel points across different coordinate systems.

A pixel $( u _ { i } , v _ { i } )$ in view i converts to 3D Cartesian coordinates $\mathbf { p } _ { i } = \left( x _ { i } , y _ { i } , z _ { i } \right)$ through $\operatorname { E q . } ( 3 )$ and $\operatorname { E q . } ( 4 )$ . The corresponding projected point in view j becomes $\mathbf { p } _ { i  j } = \mathbf { R } _ { i  j } \cdot \mathbf { p } _ { i } + \mathbf { t } _ { i  j } .$ Similarly, the camera origin $\mathbf { o } _ { i }$ projects to view j as $\mathbf { o } _ { i  j } = \mathbf { t } _ { i  j }$ The spherical epipolar line of point pi comprises projections of points along ray $\overrightarrow { \mathbf { o } _ { i } \mathbf { p } _ { i } }$ which lie on the plane II containing $\mathbf { o } _ { j } , \mathbf { p } _ { i  j }$ , and $\mathbf { o } _ { i  j } .$ For equirectangular projection, the intersection of plane I with the coordinate sphere in spherical coordinates represents the corresponding epipolar line. Expressing the plane II in the camera coordinate system of view j as $A x + B y + C z + D = 0$ , we derive coefficients through geometric constraints:

$$
\begin{array} { r l } & { A = \frac { z _ { \mathbf { 0 } _ { i  j } } \cdot y _ { \mathbf { p } _ { i  j } } - y _ { \mathbf { 0 } _ { i  j } } \cdot z _ { \mathbf { p } _ { i  j } } } { y _ { \mathbf { 0 } _ { i  j } } \cdot x _ { \mathbf { p } _ { i  j } } - x _ { \mathbf { 0 } _ { i  j } } \cdot y _ { \mathbf { p } _ { i  j } } } \cdot C = A ^ { \prime } \cdot C , } \\ & { B = \frac { z _ { \mathbf { 0 } _ { i  j } } \cdot x _ { \mathbf { p } _ { i  j } } - x _ { \mathbf { 0 } _ { i  j } } \cdot z _ { \mathbf { p } _ { i  j } } } { x _ { \mathbf { 0 } _ { i  j } } \cdot y _ { \mathbf { p } _ { i  j } } - y _ { \mathbf { 0 } _ { i  j } } \cdot x _ { \mathbf { p } _ { i  j } } } \cdot C = B ^ { \prime } \cdot C , } \\ & { D = 0 , } \end{array}\tag{7}
$$

where $( { { x _ { 0 } } _ { i  j } } , { { y _ { 0 } } _ { i  j } } , { { z _ { 0 } } _ { i  j } } )$ and $( { { x _ { { \bf { p } } } } _ { i  j } } , { { y _ { { \bf { p } } } } _ { i  j } } , { { z _ { { \bf { p } } } } _ { i  j } } )$ denote coordinates of $\mathbf { o } _ { i  j }$ and $\mathbf { p } _ { i \to j }$ respectively. Combining with the spherical constraint in Eq.(4) and converting to pixel coordinates via Eq.(3), we obtain the epipolar line parametrization:

$$
v = - \frac { H } { \pi } \left( \arctan \frac { A ^ { \prime } \sin \frac { 2 \pi u } { W } + \cos \frac { 2 \pi u } { W } } { B ^ { \prime } } \right) ,\tag{8}
$$

where (u, v) represents pixel coordinates in view $j ^ { \prime } s$ panorama.

3.3.2 Spherical Epipolar Mask. The epipolar line defines geometrically valid correspondences between source and target views by establishing plausible reference pixels. While perspective projection enables efficient distance computation through linear epipolar constraints, spherical geometry requires non-linear treatment due to curved epipolar trajectories derived from Eq.(8). We compute the minimum spherical distance between a pixel p and the epipolar line through discretized samples along the curve. To mitigate computational complexity, we uniformly sample K points $\{ \mathbf { c } _ { k } \} _ { k = 1 } ^ { K }$ along the epipolar line and approximate the minimum distance as:

$$
d _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { 1 \leq k \leq K } \| { \bf p } - { \bf c } _ { k } \| _ { 2 } .\tag{9}
$$

A pixel qualifies as a valid reference when $d _ { \mathrm { m i n } }$ falls below half the feature grid's diagonal length. This thresholding strategy ensures geometrically consistent correspondences while accommodating localization uncertainties. For each panoramic frame i in the video sequence, we compute per-pixel epipolar masks across all frames through spherical geometry constraints. The complete spherical epipolar mask $\mathbf { M } _ { i } \in \mathbb { R } ^ { H W \times \check { N } \times H W }$ is obtained by aggregating these view-consistent correspondences. As visualized in Fig. 3 (Right), the resultant binary mask restricts cross-view attention to topologically aligned regions while preserving multi-view consistency.

Fig. 4. Visualization Results of Spherical Epipolar Attention Map.

SA Conference Papers '25, December 15–18, 2025, Hong Kong, Hong Kong

Table 1. Quantitative Comparisons with Baseline Methods. Our method demonstrates significant improvements across three critical dimensions compared to baseline methods: camera view consistency, photorealistic fidelity of generated frames, and holistic video quality.
<table><tr><td rowspan="2">Method</td><td rowspan="2">LPIPS↓</td><td rowspan="2">SSIM↑</td><td rowspan="2">PSNR↑</td><td rowspan="2">FAED↓</td><td colspan="2">FVD↓</td><td colspan="3">VBench↑</td></tr><tr><td>VideoGPT</td><td>StyleGAN</td><td>Aesthetic Quality</td><td>Subject Consistency</td><td>Temporal Flickering</td></tr><tr><td>MotionCtrl [Wang et al. 2024b]</td><td>0.1741</td><td>0.6008</td><td>29.51</td><td>0.2993</td><td>94.84</td><td>73.75</td><td>0.4878</td><td>0.8706</td><td>0.9256</td></tr><tr><td>CameraCtrl [He et al. 2024]</td><td>0.1816</td><td>0.5926</td><td>29.29</td><td>0.3967</td><td>151.08</td><td>135.90</td><td>0.4832</td><td>0.8593</td><td>0.9241</td></tr><tr><td>CamI2V [Zheng et al. 2024]</td><td>0.1867</td><td>0.5887</td><td>29.24</td><td>0.2621</td><td>91.83</td><td>76.25</td><td>0.4931</td><td>0.8755</td><td>0.9302</td></tr><tr><td>CamPVG (ours)</td><td>0.1480</td><td>0.6544</td><td>30.05</td><td>0.1066</td><td>66.24</td><td>56.34</td><td>0.5043</td><td>0.9000</td><td>0.9339</td></tr></table>

3.3.3 Spherical Epipolar Attention. We introduce spherical epipolar attention to enforce multi-view consistency in panoramic video generation through explicit geometric constraints. In video diffusion models, spatial attention primarily focuses on the spatial relationships within single-frame images, while temporal attention mainly addresses the relationships between consecutive frames. Hence, we apply spherical epipolar attention before the temporal attention to facilitate the model's learning of correspondences between different viewpoints. For each query frame $q _ { i } \in \mathbb { R } ^ { H W \times C }$ , the key and value are derived from all N frames as $k \in \mathbb { R } ^ { N H W \times C }$ and $\mathsf { \Omega } _ { v } \doteq \mathbb { R } ^ { N H W \times C }$ The attention computation incorporates our precomputed spherical epipolar mask $\mathbf { M } _ { i } \overset { \bullet } { \in } \mathbb { R } ^ { H W \times N \times H ^ { \mathbf { k } } W }$ as:

$$
\mathrm { S p h e r i c E p i A t t n } ( q _ { i } , k , v ) = \mathrm { s o f t m a x } \left( \frac { q _ { i } k ^ { \top } } { \sqrt { d } } \odot \mathbf { M } _ { i } \right) v ,\tag{10}
$$

where d represents the dimension of attention heads. The visualization results of spherical epipolar attention map are shown in Fig. 4. This architectural modification enables simultaneous learning of temporal dynamics and cross-view geometric relationships, particularly crucial for maintaining 3D consistency during panoramic camera motion. The explicit geometric prior embedded in the attention mechanism guides the diffusion model to preserve scene structure across viewpoints without requiring explicit 3D reconstruction.

## 4 Experiments

## 4.1 Experiment Settings

4.1.1 Dataset. To obtain panoramic video datasets incorporating precise camera poses, we construct camera trajectories within the 3D-FRONT dataset [Fu et al. 2021]. At each position along the trajectory, we render cubemaps and convert them into panoramas through equirectangular projection. We render panoramic videos for 5,616 scenes within 3D-FRONT. We generate 40-frame sequences for each camera trajectory, and randomly sample 16 frames at a resolution of 256 × 512 to form individual video clips. Following CamI2V [Zheng et al. 2024], we implement randomized conditional frame selection as data augmentation.

4.1.2 Implementation Details. We choose DynamiCrafter [Xing et al. 2024] as our base image-to-video model, removing its text conditioning component and generating 16-frame panoramic videos. During training, we freeze all the parameters of the base model and only train our panoramic position encoder and spherical epipolar module. The spherical epipolar mask computation samples K = 250 points along each epipolar line for distance approximation. We employ the Adam optimizer with a fixed learning rate of $1 \times 1 0 ^ { - 4 }$ The model is trained on 8× NVIDIA A800 GPUs with a batch size of 16 for 300 epochs, taking approximately 4 days to complete. For fair comparison, we retrain baseline methods on our panoramic dataset with the same training settings.

4.1.3 Evaluation Metrics. As conventional pose estimation methods [Pan et al. 2024; Schönberger and Frahm 2016] for perspective videos are inapplicable to panoramic content, we assess camera trajectory consistency by comparing the fifth generated frame after conditioning inputs with corresponding ground truth frame. This temporal offset allows us to avoid minimal differences in adjacent frames and excessive generative divergence in distant frames. Frame-wise similarity is evaluated through Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM) [Wang et al. 2004], and Learned Perceptual Image Patch Similarity (LPIPS) [Zhang et al. 2018]. To evaluate the visual fidelity of generated panoramic frames, we compute the Fréchet Auto-Encoder Distance (FAED) [Zhang et al. 2024] on selected fifth-frame instances. FAED extends the Fréchet Inception Distance (FID) [Heusel et al. 2017], and is specifically designed to address equirectangular projection distortions. Additionally, we evaluate the overall quality of the panoramic video using Fréchet Video Distance (FVD) [Skorokhodov et al. 2022; Unterthiner et al. 2018; Yan et al. 2021] and VBench [Huang et al. 2024]. All metrics are computed over 1,000 randomly sampled video clips.

## 4.2 Comparisons with Baseline Methods

4.2.1 Quantitative Comparisons. As we propose the first framework for precise camera pose-guided panoramic video generation, existing methods are not directly comparable. Consequently, we adapt and retrain three perspective-domain approaches, including CameraCtrl [He et al. 2024], MotionCtrl [Wang et al. 2024b], and CamI2V [Zheng et al. 2024]. For fair comparison, we modify MotionCtrl by retaining only its camera control module while disabling object motion components. All methods utilize DynamiCrafter as the base model and are trained on our panoramic video datasets with precise camera pose annotations. As demonstrated in Tab. 1, CamPVG achieves superior performance in camera view consistency metrics (PSNR, SSIM and LPIPS), indicating more accurate reconstruction of panoramic views with less distortion and higher structural consistency.. Furthermore, our method preserves high video generation quality and visual fidelity, achieving the lowest FVD and FAED scores and the highest VBench scores, owing to the incorporation of geometric constraints. These results demonstrate that our geometric-aware constraints bridge panoramic consistency and generation fidelity.

4.2.2 Qualitative Comparisons. We present a qualitative comparison between our method and existing baseline approaches in Fig. 5. MotionCtrl [Wang et al. 2024b] controls camera viewpoints by concatenating camera poses directly with latent space features. While this approach achieves reasonable results in perspective video generation, it fails to model panoramic spatial geometry, leading to content loss and inconsistent cross-view alignment in panoramic scenarios. CameraCtrl [He et al. 2024] incorporates camera poses into the U-Net via perspective-based Plücker embeddings. However, its perspective-centric positional encoding cannot address panoramic geometric distortions, resulting in noticeable content deformation across frames. CamI2V [Zheng et al. 2024] further introduces a perspective-projection-based epipolar module. However, due to the domain gap between perspective and panoramic representations, its epipolar module fails to correctly compute the corresponding epipolar lines for target viewpoints, leading to degraded feature referencing. As a result, CamI2V struggles to preserve fine-grained details and occasionally produces content deformations. Especially in complex scenes (e.g., the second example in Fig. 5), it exhibits cross-view inconsistency and generates hallucinated content due to the absence of panoramic geometric priors. In contrast, our method explicitly models panoramic camera poses and enforces spherical epipolar constraints, ensuring high-fidelity geometric consistency across dynamically changing viewpoints. The panoramic videos generated by our approach effectively mitigate distortions while preserving intricate scene details. Even in complex scenarios with multi-room transitions, our approach maintains strict alignment with the input camera trajectory without introducing unrealistic artifacts. More generated results are shown in Fig. 7.

![](images/e2d1e1ab6430c028e935d19dd1a3922c6a1da491cc9d6651eab0207637fac9b4.jpg)  
Fig. 5. Qualitative Comparison with Baseline Methods.

![](images/10dd37bdb8ad679ee1bb0a6892462e95117d013ad477581b69a1a1c28541aa85.jpg)  
Fig. 6. Qualitative Ablation Study on Different Model Components. Removing any individual component leads to a noticeable degradation in visual quality and temporal coherence, whereas the complete model consistently achieves the best overall performance.

Table 2. Ablation Study on Different Components. We perform an ablation study evaluating the impact of removing the panoramic Plücker embedding, spherical epipolar module, and random conditional frame strategy on model performance.
<table><tr><td>Method</td><td>Pano. Plücker</td><td>Sph. Epipolar</td><td>Rand. Condition</td><td>LPIPS↓</td><td>SSIM↑</td><td>PSNR↑</td><td>FAED↓</td><td>FVD↓ VideoGPT</td><td>StyleGAN</td></tr><tr><td rowspan="5">CamPVG (ours)</td><td>✓</td><td>✓</td><td></td><td>0.1480</td><td>0.6544</td><td>30.05</td><td>0.1066</td><td>66.24</td><td>56.34</td></tr><tr><td>✓</td><td></td><td>く</td><td>0.1865</td><td>0.5998</td><td>29.48</td><td>0.2704</td><td>87.14</td><td>77.13</td></tr><tr><td></td><td>✓</td><td>✓</td><td>0.3278</td><td>0.4868</td><td>28.85</td><td>0.4954</td><td>124.93</td><td>101.03</td></tr><tr><td>く</td><td>く</td><td></td><td>0.4144</td><td>0.4794</td><td>28.51</td><td>0.7591</td><td>259.03</td><td>157.92</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 4.3 Ablation Study

4.3.1 Ablation Study on Different Components. Our method integrates geometric constraints through the panoramic Plücker embedding and the spherical epipolar module. Additionally, we employ a random conditional frame strategy during training to enhance the model's robustness. To validate the contribution of each component within CamPVG, we conduct ablation studies focusing on these three critical components. Qualitative (Fig. 6) and quantitative results (Tab. 2) are presented. The results show that removing the panoramic Plücker embedding leads to a larger performance drop than removing the spherical epipolar module. The panoramic Plücker embedding enables the model to comprehend panoramic camera pose information; without it, the model loses the ability to represent the panoramic space, leading to substantial performance degradation. In contrast, the spherical epipolar module reinforces geometric constraints across different viewpoints, thereby enhancing fine-grained consistency between generated frames. These findings indicate that the panoramic Plücker embedding is fundamental for establishing global camera pose awareness, while the spherical epipolar module complements it by ensuring cross-view geometric consistency. Additionally, to assess the impact of the random conditional frame strategy, we evaluate models trained without it by adopting a different conditional frame order during testing. As shown in the results, models trained with a fixed frame order suffer a significant performance drop across all evaluation metrics. This degradation is caused by overfitting to the specific order observed during training, which limits the model's ability to adapt when the order is altered at inference time. These results demonstrate that the random conditional frame strategy is crucial for enhancing both the robustness and generalization capability of the model.

Table 3. Ablation Study on Sampling Density. Both insufficient and excessive numbers of sampling points degrade model performance, with the best performance observed when K = 250.
<table><tr><td rowspan="2">Number of K</td><td rowspan="2">LPIPS↓</td><td rowspan="2">SSIM↑</td><td rowspan="2">PSNR↑</td><td rowspan="2">FAED↓</td><td colspan="2">FVD↓</td></tr><tr><td>VideoGPT</td><td>StyleGAN</td></tr><tr><td> $K = 1 0 0$ </td><td>0.1525</td><td>0.6398</td><td>29.91</td><td>0.1240</td><td>72.68</td><td>63.78</td></tr><tr><td> $K = 1 5 0$ </td><td>0.1499</td><td>0.6429</td><td>29.98</td><td>0.1146</td><td>71.22</td><td>63.56</td></tr><tr><td> $K = 2 0 0$ </td><td>0.1500</td><td>0.6457</td><td>29.94</td><td>0.1141</td><td>67.66</td><td>57.63</td></tr><tr><td> $K = 2 5 0$ </td><td>0.1480</td><td>0.6544</td><td>30.05</td><td>0.1066</td><td>66.24</td><td>56.34</td></tr><tr><td> $K = 3 0 0$ </td><td>0.1486</td><td>0.6458</td><td>29.99</td><td>0.1130</td><td>75.43</td><td>68.37</td></tr></table>

Table 4. User Study. More participants prefer the panoramic videos generated by our CamPVG. The right two columns show the comparison results under real-world inputs. Our method achieves higher preference rates across all metrics.
<table><tr><td>Method</td><td>Camera Consistency↑</td><td>Condition Consistency↑</td><td>Video Quality↑</td><td>Condition Consistency↑</td><td>Video Quality↑</td></tr><tr><td>MotionCtrl [Wang et al. 2024b]</td><td>1.895</td><td>1.898</td><td>1.788</td><td>1.483</td><td>1.500</td></tr><tr><td>CameraCtrl [He et al. 2024]</td><td>1.835</td><td>1.743</td><td>1.895</td><td>1.927</td><td>1.877</td></tr><tr><td>CamI2V [Zheng et al. 2024]</td><td>2.753</td><td>2.763</td><td>2.770</td><td>2.750</td><td>2.850</td></tr><tr><td>CamPVG (ours)</td><td>3.518</td><td>3.598</td><td>3.548</td><td>3.817</td><td>3.783</td></tr></table>

4.3.2 Ablation Study on Sampling Density. As discussed in Sec. 3.3.2, spherical epipolar lines are characterized by uniformly sampling points along the curve. The number of sampling points significantly impacts model performance. Insufficient sampling may miss valid correspondences due to large intervals between samples, while excessive sampling introduces noise from invalid references. To determine the optimal sampling density, we conduct ablation experiments with K ∈ {100, 150, 200, 250, 300}, while keeping the width of the generated videos at 512 pixels. As shown in Tab. 3, the model achieves the best results in camera view consistency, photorealistic fidelity, and overall video quality when K = 250. Moreover, increasing the number of reference points to K = 300 leads to performance degradation due to an excess of irrelevant points. These findings emphasize the necessity of carefully selecting an appropriate K to optimize the model's ability to capture essential geometric details.

## 4.4 User Study

To complement our quantitative comparison, we conduct a human evaluation to compare our CamPVG against baseline methods (MotionCtrl, CameraCtrl, and CamI2V). For each method, we generate 20 panoramic video sequences using identical conditional frames. We invited 20 volunteers to evaluate the generated panoramic videos across three dimensions: camera trajectory consistency, consistency with the conditional images, and overall video quality. Participants rate each aspect on a scale from 1 to 4, with higher scores indicating better performance. To further evaluate the generalization capability of our method, we additionally select real-world panoramic images as conditional inputs and conduct an extended user study. As shown in Tab. 4, CamPVG achieves superior ratings across all evaluation dimensions, consistently outperforming the baseline methods. Notably, even under real-world inputs, our method attains the highest performance (see the rightmost two columns of the table). These results highlight the effectiveness of CamPVG. Additional qualitative results in diverse scenarios are provided in the supplementary material.

## 5 Conclusion

In this work, we propose CamPVG, the first diffusion-based framework for panoramic video generation guided by precise camera poses. By introducing panoramic Plücker embedding with the pose encoder, our method effectively learns panoramic camera geometry, enabling more accurate modeling of camera trajectories based on panoramic images. Additionally, through the spherical epipolar module, we achieve fine-grained feature aggregation by leveraging features along epipolar lines from different viewpoints, thereby enhancing consistency across video frames. Compared to other camera-controlled video generation methods, our approach demonstrate state-of-the-art performance in panoramic video generation, excelling in camera trajectory consistency, frame realism, and overall video quality.

Limitation. Our method is currently limited by the availability of panoramic datasets with accurate camera pose annotations, which affects its performance in complex outdoor environments. Due to these dataset constraints, the generated panoramic videos are primarily from static scenes.

Future Work. In future work, we plan to incorporate dynamicscene panoramic data to enhance motion realism and improve the generalization ability of our framework.

## Acknowledgments

This work was supported by National Natural Science Fund of China (No.62473286).

![](images/e86b7cb85a409cf00d9a059c5b9885d62aa229b56b9ff1f648d40eee9a43b3b2.jpg)  
Fig. 7. More Generated Results of CamPVG.  
SA Conference Papers '25, December 15–18, 2025, Hong Kong, Hong Kong.

## References

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. 2023. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. CoRR abs/2311.15127 (2023). doi:10.48550/ARXIV.2311. 15127 arXiv:2311.15127

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. 2024. Video generation models as world simulators. (2024). https: //openai.com/research/video-generation-models-as-world-simulators

Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. 2023. VideoCrafter1: Open Diffusion Models for High-Quality Video Generation. CoRR abs/2310.19512 (2023). doi:10.48550/ARXIV.2310.19512 arXiv:2310.19512

Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. 2024. VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024. IEEE, 7310–7320. doi:10. 1109/CVPR52733.2024.00698

Huan Fu, Bowen Cai, Lin Gao, Ling-Xiao Zhang, Jiaming Wang, Cao Li, Qixun Zeng, Chengyue Sun, Rongfei Jia, Binqiang Zhao, et al. 2021. 3d-front: 3d furnished rooms with layouts and semantics. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 10933-10942.

Junyao Gao, Jiaxing Li, Wenran Liu, Yanhong Zeng, Fei Shen, Kai Chen, Yanan Sun, and Cairong Zhao. 2025a. CharacterShot: Controllable and Consistent 4D Character Animation. arXiv preprint arXiv:2508.07409 (2025).

Junyao Gao, Yanchen Liu, Yanan Sun, Yinhao Tang, Yanhong Zeng, Kai Chen, and Cairong Zhao. 2024. Styleshot: A snapshot on any style. arXiv preprint arXiv:2407.01414 (2024).

Junyao Gao, Yanan Sun, Fei Shen, Xin Jiang, Zhening Xing, Kai Chen, and Cairong Zhao. 2025b. Faceshot: Bring any character into life. arXiv preprint arXiv:2503.00740 (2025).

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. 2024. CameraCtrl: Enabling Camera Control for Text-to-Video Generation. CoRR abs/2404.02101 (2024). doi:10.48550/ARXIV.2404.02101 arXiv:2404.02101

Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. 2022. Latent Video Diffusion Models for High-Fidelity Video Generation with Arbitrary Lengths. CoRR abs/2211.13221 (2022). doi:10.48550/ARXIV.2211.13221 arXiv:2211.13221

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (Eds.). 6626-6637. https://proceedings.neurips.cc/paper/2017/hash/ 8a1d694707eb0fefe65871369074926d-Abstract.html

Jonathan Ho, Tim Salimans, Alexey A. Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. 2022. Video Diffusion Models. CoRR abs/2204.03458 (2022). doi:10.48550/ARXIV.2204.03458 arXiv:2204.03458

Teng Hu, Jiangning Zhang, Ran Yi, Yating Wang, Hongrui Huang, Jieyu Weng, Yabiao Wang, and Lizhuang Ma. 2024. MotionMaster: Training-free Camera Motion Transfer For Video Generation. CoRR abs/2404.15789 (2024). doi:10.48550/ARXIV.2404.15789 arXiv:2404.15789

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. 2024. VBench: Comprehensive Benchmark Suite for Video Generative Models. In CVPR. IEEE, 21807–21818.

Yash Jain, Anshul Nasery, Vibhav Vineet, and Harkirat S. Behl. 2024. Peekaboo: Interactive Video Generation via Masked-Diffusion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024. IEEE, 8079-8088. doi:10.1109/CVPR52733.2024.00772

Xin Jiang, Hao Tang, Junyao Gao, Xiaoyu Du, Shengfeng He, and Zechao Li. 2024b. Delving into multimodal prompting for fine-grained visual classification. In Proceedings of the AAAI conference on artificial intelligence, Vol. 38. 2570–2578.

Xin Jiang, Hao Tang, and Zechao Li. 2024a. Global meets local: Dual activation hashing network for large-scale fine-grained image retrieval. IEEE Transactions on Knowledge and Data Engineering 36, 11 (2024), 6266–6279.

Jing Yu Koh, Honglak Lee, Yinfei Yang, Jason Baldridge, and Peter Anderson. 2021. Pathdreamer: A World Model for Indoor Navigation. In ICCV. IEEE, 14718–14728.

Zhengfei Kuang, Shengqu Cai, Hao He, Yinghao Xu, Hongsheng Li, Leonidas J. Guibas, and Gordon Wetzstein. 2024. Collaborative Video Diffusion: Consistent Multi-video Generation with Camera Control. In NeurIPS

Renjie Li, Panwang Pan, Bangbang Yang, Dejia Xu, Shijie Zhou, Xuanyang Zhang, Zeming Li, Achuta Kadambi, Zhangyang Wang, and Zhiwen Fan. 2024. 4K4DGen: Panoramic 4D Generation at 4K Resolution. CoRR abs/2406.13527 (2024). doi:10. 48550/ARXIV.2406.13527 arXiv:2406.13527

Jinxiu Liu, Shaoheng Lin, Yinxiao Li, and Ming-Hsuan Yang. 2025. DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes. In CVPR. Computer Vision Foundation / IEEE, 6144–6153.

Linfei Pan, Daniel Barath, Marc Pollefeys, and Johannes Lutz Schönberger. 2024. Global Structure-from-Motion Revisited. In European Conference on Computer Vision (ECCV).

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-Resolution Image Synthesis with Latent Diffusion Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, 7une 18-24, 2022. IEEE, 10674–10685. doi:10.1109/CVPR52688.2022.01042

Johannes L. Schönberger and Jan-Michael Frahm. 2016. Structure-from-Motion Revisited. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016. IEEE Computer Society, 4104–4113. doi:10.1109/CVPR.2016.445

Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh Tenenbaum, and Frédo Durand. 2021. Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, Marc'Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan (Eds.). 19313-19325. https://proceedings. neurips.cc/paper/2021/hash/a11ce019e96a4c60832eadd755a17a58-Abstract.html

Ivan Skorokhodov, Sergey Tulyakov, and Mohamed Elhoseiny. 2022. StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, 7une 18-24, 2022. IEEE, 3616–3626. doi:10.1109/CVPR52688.2022. 00361

Jing Tan, Shuai Yang, Tong Wu, Jingwen He, Yuwei Guo, Ziwei Liu, and Dahua Lin. 2024. Imagine360: Immersive 360 Video Generation from Perspective Anchor. CoRR abs/2412.03552 (2024). doi:10.48550/ARXIV.2412.03552 arXiv:2412.03552

Kexian Tang, Junyao Gao, Yanhong Zeng, Haodong Duan, Yanan Sun, Zhening Xing, Wenran Liu, Kaifeng Lyu, and Kai Chen. 2025. LEGO-Puzzles: How Good Are MLLMs at Multi-Step Spatial Reasoning? arXiv preprint arXiv:2503.19990 (2025).

Shitao Tang, Fuyang Zhang, Jiacheng Chen, Peng Wang, and Yasutaka Furukawa. 2023. MVDiffusion: Enabling Holistic Multi-view Image Generation with Correspondence-Aware Diffusion. In NeurIPS.

Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. 2018. Towards Accurate Generative Models of Video: A New Metric & Challenges. CoRR abs/1812.01717 (2018). arXiv:1812.01717 http //arxiv.org/abs/1812.01717

Qian Wang, Weiqi Li, Chong Mou, Xinhua Cheng, and Jian Zhang. 2024a. 360DVD: Controllable Panorama Video Generation with 360-Degree Video Diffusion Model. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, 7une 16-22, 2024. IEEE, 6913-6923. doi:10.1109/CVPR52733.2024.00660

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13, 4 (2004), 600–612.

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. 2024b. MotionCtrl: A Unified and Flexible Motion Controller for Video Generation. In ACM SIGGRAPH 2024 Conference Papers, SIGGRAPH 2024, Denver, CO, USA, 27 July 2024- 1 August 2024, Andres Burbano, Denis Zorin, and Wojciech Jarosz (Eds.). ACM, 114. doi:10.1145/3641519.3657518

Kevin Xie, Amirmojtaba Sabour, Jiahui Huang, Despoina Paschalidou, Greg Klar, Umar Iqbal, Sanja Fidler, and Xiaohui Zeng. 2025. VideoPanda: Video Panoramic Diffusion with Multi-view Attention. CoRR abs/2504.11389 (2025).

Jinbo Xing, Menghan Xia, Yong Zhang, Hao Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. 2024. DynamiCrafter: Animating Open-Domain Images with Video Diffusion Priors. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XLVI (Lecture Notes in Computer Science, Vol. 15104), Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol (Eds.). Springer, 399-417. doi:10.1007/978-3-031-72952-2 23

Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, and Arash Vahdat. 2024. CamCo: Camera-Controllable 3D-Consistent Image-to-Video Generation. CoRR abs/2406.02509 (2024). doi:10.48550/ARXIV.2406.02509 arXiv:2406.02509

Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. 2021. VideoGPT: Video Generation using VQ-VAE and Transformers. CoRR abs/2104.10157 (2021). arXiv:2104.10157https://arxiv.org/abs/2104.10157

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, and Jie Tang. 2024. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer CoRR abs/2408.06072 (2024). doi:10.48550/ARXIV.2408.06072 arXiv:2408.06072

Weicai Ye, Chenhao Ji, Zheng Chen, Junyao Gao, Xiaoshui Huang, Song-Hai Zhang, Wanli Ouyang, Tong He, Cairong Zhao, and Guofeng Zhang. 2024 DiffPano: Scalable and Consistent Text to Panorama Generation with Spherical Epipolar-Aware Diffusion. In Advances in Neural Information Processing

Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (Eds.). http://papers.nips.cc/paper\_files/paper/2024/hash/ 02c1d1d33dbfbaf03b3971bb542e72e2-Abstract-Conference.html

Xiaoding Yuan, Shitao Tang, Kejie Li, and Peng Wang. 2025. CamFreeDiff: Camera-free Image to Panorama Generation with Diffusion Model. In CVPR. Computer Vision Foundation / IEEE, 16408-16417.

Cheng Zhang, Qianyi Wu, Camilo Cruz Gambardella, Xiaoshui Huang, Dinh Phung, Wanli Ouyang, and Jianfei Cai. 2024. Taming Stable Diffusion for Text to 360o

Panorama Image Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. 2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018. Computer Vision Foundation / IEEE Computer Society, 586-595. doi:10.1109/CVPR.2018.00068

Guangcong Zheng, Teng Li, Rui Jiang, Yehao Lu, Tao Wu, and Xi Li. 2024. CamI2V: Camera-Controlled Image-to-Video Diffusion Model. CoRR abs/2410.15957 (2024). doi:10.48550/ARXIV.2410.15957 arXiv:2410.15957