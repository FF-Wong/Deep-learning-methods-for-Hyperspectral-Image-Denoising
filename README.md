# Deep-learning-methods-for-Hyperspectral-Image-Denoising
在 [Yongsen Zhao]( https://github.com/seniusen) 和 [Junjun Jiang](http://homepage.hit.edu.cn/jiangjunjun)整理的高光谱遥感影像去噪方法清单[Hyperspectral-Image-Denoising-Benchmark](https://github.com/junjun-jiang/Hyperspectral-Image-Denoising-Benchmark)的基础上，进行了深度学习方法的更新。

#### 基于深度学习的高光谱去噪方法
- Hyperspectral imagery denoising by deep learning with trainable nonlinearity function, GRSL 2017, W. Xie et al.
- [HSID-CNN]:Hyperspectral Image Denoising Employing a Spatial-Spectral Deep Residual Convolutional Neural Network, TGRS 2018, Q. Yuan et al. [[Code]](https://github.com/WHUQZhang/HSID-CNN)
- [HSI-DeNet]: Hyperspectral Image Restoration via Convolutional Neural Network, TGRS 2019, Yi Chang et al. [[Web]](http://www.escience.cn/people/changyi/index.html) [[Pdf]](http://www.escience.cn/system/download/100951)
- [Deep Hyperspectral Prior]: Denoising, Inpainting, Super-Resolution, arxiv 2019, Oleksii Sidorov et al. [[Code]](https://github.com/acecreamu/deep-hs-prior) [[Pdf]](https://arxiv.org/pdf/1902.00301)
- [SSGN]:Hybrid Noise Removal in Hyperspectral Imagery With a Spatial-Spectral Gradient Network, IEEE TGRS 2019, Qiang Zhang et al. [[Code]](https://github.com/WHUQZhang/SSGN) [[Pdf]](https://arxiv.org/pdf/1810.00495)
- Deep Spatial-spectral Representation Learning for Hyperspectral Image Denoising, IEEE TCI 2019, Weisheng Dong et al. 
- Hyperspectral Image Denoising via Matrix Factorization and Deep Prior Regularization,IEEE TIP 2020,Baihong Lin et al[[Pdf]](https://doi.org/10.1109/TIP.2019.2928627)
- [HSI-SDeCNN]:A Single Model CNN for Hyperspectral,IEEE TGRS 2020(Early Access),Alessandro Maffei et al. [[Code]](https://github.com/mhaut/HSI-SDeCNN) [[Pdf]](https://doi.org/10.1109/TGRS.2019.2952062)
- Toward Universal Stripe Removal via Wavelet-Based Deep Convolutional Neural Network,IEEE TGRS 2020,Yi Chang et al.
- Deep spatio-spectral Bayesian posterior for hyperspectral image non-i.i.d. noise removal, JPRS 2020,Qiang Zhang et al. [[Pdf]](https://www.researchgate.net/profile/Qiang_Zhang204/publication/340988173_Deep_spatio-spectral_Bayesian_posterior_for_hyperspectral_image_non-iid_noise_removal/links/5eaa25f6a6fdcc70509afdfd/Deep-spatio-spectral-Bayesian-posterior-for-hyperspectral-image-non-iid-noise-removal.pdf)


#### Databases 
- [CAVE dataset](http://www.cs.columbia.edu/CAVE/databases/multispectral/)
- [AVIRIS](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
- [ROSIS](http://lesun.weebly.com/hyperspectral-data-set.html)
- [HYDICE](https://www.erdc.usace.army.mil/Media/Fact-Sheets/Fact-Sheet-Article-View/Article/610433/hypercube/)
- [EO-1 Hyperion Data](https://lta.cr.usgs.gov/ALI)
- [Harvard dataset](http://vision.seas.harvard.edu/hyperspec/explore.html)
- [iCVL dataset](http://icvl.cs.bgu.ac.il/hyperspectral/)
- [NUS datase](https://sites.google.com/site/hyperspectralcolorimaging/dataset/general-scenes)
- [NTIRE18 dataset](http://www.vision.ee.ethz.ch/ntire18/)


#### Image Quality Measurement 
- Peak Signal to Noise Ratio (PSNR)
- Structural SIMilarity index (SSIM)
- Feature SIMilarity index (FSIM)
- Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS)
- Spectral Angle Mapper (SAM)


#### Reference
- [高光谱图像重构常用评价指标及其Python实现](https://www.cnblogs.com/nwpuxuezha/p/6659153.html)
- [Hyperspectral-Image-Denoising-Benchmark](https://github.com/junjun-jiang/Hyperspectral-Image-Denoising-Benchmark)
- [遥感学报公众号 高光谱遥感数据集](https://mp.weixin.qq.com/s?__biz=MzU2MTM4MTYzOQ==&mid=2247489064&idx=1&sn=41f2ab5c13a52dac6fe0064ae017c3a8&chksm=fc78fe40cb0f7756143a045e3e97beffac330e35fa524f82ac6869613f4cf6720d29b497b915&mpshare=1&scene=23&srcid=0327fl4R2j2zrrvURZGMHGXN&sharer_sharetime=1585311603251&sharer_shareid=5ef37c06898efb1fdf6df98cdb7ba765#rd)
