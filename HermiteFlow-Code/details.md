### **Modular Architecture of Feature-Conditioned Hermite-VFI**

#### **1. The Head: Feature & Tangent Extraction (Pre-trained Neural Network)**
* **Module:** Flow Estimator (e.g., RAFT).
* **Input:** Raw original frames $I_0, I_1 \in \mathbb{R}^{3 \times H \times W}$.
* **Action:** Analyzes the images to extract baseline velocity tangents and context.
* **Outputs:** * Bidirectional raw optical flow vectors: $F_{0\rightarrow1}, F_{1\rightarrow0} \in \mathbb{R}^{2 \times H \times W}$
    * Deep latent contextual features: $A_0, A_1 \in \mathbb{R}^{C \times H/4 \times W/4}$

#### **2. The Brains: Latent Coefficient Predictor (Shallow CNN)**
* **Module:** Compact Convolutional Network.
* **Input:** Stacked raw flows ($F_{0\rightarrow1}, F_{1\rightarrow0}$) and deep features ($A_0, A_1$).
* **Action:** Functions as the latent-space decision maker. It evaluates context (edges, occlusions, sudden stops) to recognize where strict physics equations must be customized.
* **Outputs:** Dense, pixel-by-pixel Hermite polynomial coefficients: $\alpha(x,y), \beta(x,y), \gamma(x,y), \delta(x,y) \in \mathbb{R}^{4 \times H \times W}$.

#### **3. The Engine: Pixel-Space Hermite Flow (Pure Math)**
* **Module:** Algebraic Cubic Spline Equations.
* **Input:** $F_{0\rightarrow1}, F_{1\rightarrow0}$, predicted dense coefficients, and target time scalar $t \in (0, 1)$.
* **Action:** Calculates the exact, context-aware continuous trajectory for every pixel in $O(1)$ time. The polynomial dynamically adapts to use beautiful, smooth curves for regular motion, or automatically dampens into linear trajectories near occlusion boundaries to prevent ghosting artifacts.
* **Physics:** The Physics Mapping1. Start Position ($P_0$):As we agreed, in relative space, the pixel acts as its own origin.$$P_0 = 0$$2. Start Velocity ($V_0$):This is $dx/dt$ at $t=0$. In video frames, $\Delta t = 1$ frame. So your velocity is exactly your optical flow vector pointing to Frame 1.$$V_0 = F_{0\rightarrow1}$$3. End Position ($P_1$):This is where you got confused by my previous explanation! Since the pixel started at $0$, and it traveled with a total displacement of $F_{0\rightarrow1}$ over $1$ frame, its final position $P_1$ is exactly equal to $F_{0\rightarrow1}$. *$$P_1 = 0 + F_{0\rightarrow1} = F_{0\rightarrow1}$$(Think of it this way: If you start at mile marker $0$ ($P_0=0$), and you drive $60$ miles per hour ($F=60$) for $1$ hour ($t=1$), your ending position $P_1$ is mile marker $60$. Position and Velocity share the exact same number in this specific 1-timestep scenario!)4. End Velocity ($V_1$):This is $dx/dt$ as the pixel arrives at $t=1$. RAFT gives us the backward flow $F_{1\rightarrow0}$ (pointing back in time). To make it a forward-facing velocity tangent for our math, we just multiply by $-1$.$$V_1 = -F_{1\rightarrow0}$$Plugging it into the Hermite EquationNow, let's look at the standard Cubic Hermite formula:$$P(t) = h_{00}(t)P_0 + h_{10}(t)V_0 + h_{01}(t)P_1 + h_{11}(t)V_1$$Substitute our four relative variables into it:$$P(t) = h_{00}(t)\mathbf{(0)} + h_{10}(t)\mathbf{(F_{0\rightarrow1})} + h_{01}(t)\mathbf{(F_{0\rightarrow1})} + h_{11}(t)\mathbf{(-F_{1\rightarrow0})}$$Because $P_0 = 0$, that first term drops out entirely. But the $P_1$ term stays! We can actually group the math to make it incredibly clean:$$P(t) = \Big[h_{10}(t) + h_{01}(t)\Big] F_{0\rightarrow1} - \Big[h_{11}(t)\Big] F_{1\rightarrow0}$$

* **Outputs:** Bilateral intermediate flows pointing to time $t$: $F_{t\rightarrow0}, F_{t\rightarrow1} \in \mathbb{R}^{2 \times H \times W}$.

#### **4. The Canvas: Backward Warping (Pure Math)**
* **Module:** Bilinear Interpolation Operator ($\overline{\omega}$).
* **Input:** $I_0, I_1$ and calculated intermediate flows ($F_{t\rightarrow0}, F_{t\rightarrow1}$).
* **Action:** Looks backward to physically pull the original pixel colors into their new coordinates, constructing draft images.
* **Outputs:** Physically-grounded draft images: $I_{t\rightarrow0}, I_{t\rightarrow1} \in \mathbb{R}^{3 \times H \times W}$.

#### **5. The Tail: Synthesis & Masking (Lightweight CNN)**
* **Module:** UNet-Style Synthesis Decoder.
* **Input:** Stacked draft images ($I_{t\rightarrow0}, I_{t\rightarrow1}$), intermediate flows, and original latent features ($A_0, A_1$).
* **Action:** Acts as the high-frequency visual safety net. It uses high-dimensional latent context to identify sharp object edges that pixel-space math might have stretched or torn, seamlessly stitching foreground object seams over background layers.
* **Outputs:** * Occlusion blending mask: $M_t \in \mathbb{R}^{1 \times H \times W}$
    * Final interpolated frame: $\hat{I}_t = M_t \odot I_{t\rightarrow0} + (1 - M_t) \odot I_{t\rightarrow1}$

![1782224687875](image/details/1782224687875.png)