# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 23:23:23 2022

@author: mahdi
"""


# In[3]:


from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import random
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42



# In[11
# 3]:


def unit_vector(indx, size_v):
    ### Generates a vector which all elements are zero except its i-th index
    out=np.zeros(size_v)
    out[indx]=1
    return out
    


# In[148]:




# In[13]:


def shifter(shift, size):
    ### Look at the following example 
    I1=np.eye(size-shift)
    I2=np.eye(shift)
    out=np.zeros([size, size])
    out[shift:, 0:size-shift]=I1
    out[0:shift,size-shift:]=I2
    return out
   


# In[16]:


shifter(1,6)


# In[ ]:


def distance(i, j, n):
    val1 = abs(i - j)
    val2 = abs(i + n - j)
    val3 = abs(i - n - j)
    return min(val1, val2, val3)
    


# In[99]:


def gen_h(peak_index, size):
    h=np.matmul(shifter(peak_index - 1, size), np.reshape([abs(size/2 - i) for i in range(size)],[size,-1])) +np.reshape(size / 5 * np.random.rand(size),[size,-1])
    h=np.reshape(h,[1,size])
    h=h[0]
    h -= np.min((h)) * np.ones(size)
    h /= 1.25 *np. max((h))
    h[peak_index] = 1
    return h


def gen_h_sequence(peak_index, size, number_sequence):
    ###generating a random signal for number_sequence sequences 
    h_ls=[]
    peak = peak_index
    h=gen_h(peak_index ,size)
    h_ls.append(h)
    for indx in range(number_sequence):
        # if indx%5==0:
        
        #     rand=np.random.rand()
        #     if rand<0.5:
        #         shift_peak_indx=1
        #     else:
        #         shift_peak_indx=-1
        #     # peak = peak + peak+shift_peak_indx
        #     peak += shift_peak_indx
        # h = gen_h(np.mod(peak, n), n)
        h_ls.append(h)

    return h_ls
# In[108]:


"""
v

value(g1) * n = 9.977457450939115
value(g2) = 0.9998872872546984
value(g3) = 80.0
value(mu1) = -0.19774574509396825
"""


# In[143]:


g=[9.977457450939115,0.9998872872546984, 80 ]


# In[126]:



# In[ ]:
def Neural_filter_update(h,v_old, v_sum_old, p):
    g=[9.977457450939115,0.9998872872546984, 80 ]
   
    dt=p['time_step']
    n=int(p['bin_size']/p['nmbr_peaks'])
    g1=g[0]/n
    g2=g[1]
    g3=g[2]
    v=np.zeros_like(v_old)
    for j in range(n):
        dv=-v_old[j] + max(0, h[j] - g1 * v_sum_old + g2 * v_old[j])
        v_sum_truth = sum(v_old)
        v_sum = v_sum_truth + (v_sum_old - v_sum_truth) * np.exp(-g3 * dt) 
        v [j]= v_old[j] + dt * dv
    return v, v_sum

def Neural_filter(h_ls, p):
  
    
    
    v_ls=[]
    vsum_ls=[]
    n=p['bin_size']
    # parts = np.linspace(0,n, num=p['nmbr_peaks'])
    
    v=h_ls[0]
    v_part = np.reshape(v, [p['nmbr_peaks'],-1])
    v_sum_part = np.sum(v_part, axis = -1)
    v_sum=sum(v)
    v_ls.append(v)
    vsum_ls.append(v_sum)

    for indx,h in enumerate(h_ls):
        if indx==0:
            continue
        else:
            h_part = np.reshape(h,[p['nmbr_peaks'],-1])
            v_new=[]
            v_sum_new=[]
            for i in range(p['nmbr_peaks']):
                
                v,v_sum= Neural_filter_update(h_part[i], v_part[i], v_sum_part[i], p)
                v_new.append(v)
                v_sum_new.append(v_sum)
           
            v_part = v_new
            v_sum_part = v_sum_new
                
                
                
            v,v_sum= Neural_filter_update(h, v, v_sum, p)
            v_ls.append(np.reshape(np.array(v_part), [1,-1])[0])
            vsum_ls.append(v_sum)
    return v_ls, vsum_ls            



p={'bin_size':24,
   'time_step':0.2,
   'peak_index':10,
   'nmbr_iterations':200,
   'nmbr_peaks': 6
  }


dt=p['time_step']

n=p['bin_size']
g1=g[0]/n
g2=g[1]
g3=g[2]


h_ls=gen_h_sequence(p['peak_index'], p['bin_size'], p['nmbr_iterations'])
for ix in range(len(h_ls)):
       
        h_ls[ix] = 0.4*h_ls[ix]
        
        h_ls[ix][10] = 1
        h_ls[ix][5]=0.7
        h_ls[ix][16]=0.6

        if ix>150:
            h_ls[ix] = 0.1 *np.ones_like(h_ls[ix])
# # # h_ls=np.load('out_sample.npy')
# # # real_speed = np.load('label_sample.npy')
# # # real_speed=np.reshape(real_speed,[len(real_speed), p['bin_size']])
v_ls,_= Neural_filter(h_ls, p)





fig,ax= plt.subplots()
for i in range( p['nmbr_iterations']):
    sc = 1
    wd = 0.4
    ind = np.arange(len(v_ls[i]))
    ax.bar(ind-wd/2,v_ls[i], width=wd, alpha=1,ls='solid', color='blue')
    ax.bar(ind+wd/2,h_ls[i],width=wd, ls='solid',  alpha=1, color='red')
    # ax.plot(sc*v_ls[i], color='blue', alpha=0.7)
    # ax.plot(h_ls[i], color='red',alpha=0.7)
    ax.legend(['Output', 'Input'])
    
    
    fname = 'res_filter'
    


    # ax[1].plot(v_ls[i], color='blue')
    # ax[1].legend('Filter')
    # ax[2].plot(h_ls[i], color='red')
    # ax[2].legend('Neural Network')
    # ax[3].plot(real_speed[i], color= 'green')
    # ax[3].legend('Real Speed')
    fig.savefig(fname+'/'+'f'+str(i)+'.png')
    
    
    ax.clear()
    ax.clear()


        

# %%
