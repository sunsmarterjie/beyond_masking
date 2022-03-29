# beyond_masking
**[Beyond Masking: Demystifying Token-Based Pre-Training for Vision Transformers](https://arxiv.org/pdf/2203.14313.pdf)**


<p align="center">
  <img src="img/pipeline.png" alt="beyond masking" width="70%">
</p>
<p align="center">
Figure 1: Pipeline of token-based pre-training.
</p>

<p align="center">
  <img src="img/vis.png" alt="visualization" width="60%">
</p>
<p align="center">
Figure 2: The visualization of the proposed 5 tasks.
</p>

** main results**

All the results are pre-trained for 300 epochs using Vit-base as default.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">zoomed-in</th>
<th valign="bottom">zoomed-out</th>
<th valign="bottom">distorted</th>
<th valign="bottom">blurred</th>
<th valign="bottom">de-colorized</th>
<!-- TABLE BODY -->
<tr><td align="left">FT</td>
<td align="center"><tt>82.7</tt></td>
<td align="center"><tt>82.5</tt></td>
<td align="center"><tt>82.1</tt></td>
<td align="center"><tt>81.8</tt></td>
<td align="center"><tt>81.4</tt></td>
</tr>
</tbody></table>

The code will come before 3.31
