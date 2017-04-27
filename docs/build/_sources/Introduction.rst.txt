Introduction to IDEM
=====================

The Inductive Design Exploration Method is a tool for finding robust solutions to multi-level design problems with input parameters at each level of interest.
Each level, :math:`k`, has :math:`m` objective functions, resulting in a :math:`m`-dimensional space of feasible solutions.
Starting at the top level, :math:`k=l`, three steps are completed to find the feasible region at the level directly below, :math:`k-1`:

1. At the :math:`k-1` level, multiple input parameter configurations :math:`\bar{x}` are sampled.
2. Each :math:`\bar{x}` is then projected onto the :math:`k`-level depending on the input and mapping function uncertainties, resulting in an output range :math:`\bar{y}` at level :math:`k`.
3. The :math:`k`-level output range :math:`\bar{y}` resulting from :math:`\bar{x}` of level :math:`k-1` is accepted or rejected according to an error margin which define the feasible region of level :math:`k`.

The determined feasible input parameter configurations of the :math:`k-1` level are subsequently used as the new acceptable output range to find the :math:`k-2` level design feasible inputs.

Further details
================
The output range :math:`\bar{y}` is represented by :math:`\langle f_1(\bar{x}),f_2(\bar{x}),\ldots,f_m(\bar{x})\rangle`, where :math:`m` is the number of output functions.
Each output function :math:`f_i` has an associated range of uncertainty as a result of the projection process, which can be bounded to incorporate that uncertainty using a set of functions :math:`f_{i,lower} (x) \leq f_i (x) \leq f_{i,upper} (x)`.
This results in the construction of an :math:`m \times 3` matrix :math:`\mathbf{Z}`, where component :math:`z_{ij}` is the :math:`j^{th}` bounding function of the :math:`i^{th}` output dimension.

Using these output ranges and bounding functions, a total assumed variability for the :math:`i^{th}` dimension can be found, :math:`\Delta y_i` from:

.. math::
   
   \Delta y_i = \left\{ \begin{array}{cc} \left| max_j \left( z_{ij} (\bar{x}) + \left| \frac{\delta z_{ij}}{\delta x_k} \right| \right) - \bar{y} \right| , b_i > \bar{y}_i \\ \left| min_j \left( z_{ij} (\bar{x}) - \left| \frac{\delta z_{ij}}{\delta x_k} \right| \right) - \bar{y} \right| , b_i \leq \bar{y}_i \end{array}\right.
   
where :math:`b_i` is the boundary point being searched for on :math:`b_i \in S` where :math:`S` is the :math:`(m-1)`-dimensional boundary of feasible space :math:`\Omega`.
pyDEM finds :math:`b_i` along the path :math:`\langle \bar{y}, max(S_i) \rangle` using a binary search, to minimize the distance required to reach the boundary.

Error margins
==============

Hyper-Dimensional Error Margin Index :math:`(HD_{EMI})`
---------------------------------------------------------
.. math::

   HD_{EMI} = \left\{ \begin{array}{cc} min_i \left[ \frac{\left\Vert \bar{y}_i - b_i \right\Vert}{\Delta y_i} \right] , & \mbox{for} & \bar{y} \in \Omega \\ -1 , & \mbox{for} & \bar{y} \notin \Omega \end{array}\right.
   
A value of :math:`HD_{EMI} < 1` indicates the output range of the potential solution falls outside the feasible range and is not a robust solution.

Valid Output Region :math:`VOR` 
-------------------------------
Checks that the entire bounds of the output region lie within the feasible region of input uncertainty.

Maximum Independent Variation :math:`MIV` 
------------------------------------------
Varies a single output parameter at a time, similar to :math:`HD_{EMI}`.