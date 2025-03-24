def pcorr4df(df=None, cosineThrd=0, imshow=False, dobokeh=False, bokehSquareSize=1):
    import pyspark
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.sql import functions as F
    # from pyspark.mllib.linalg import Vectors   
    import numpy as np 
    import matplotlib.pyplot as plt
    
    # display(df.describe()) #.show()
    dfs=df.describe().toPandas()
    #demean such that cosine become Pearson correlation
    labels=list(df.columns)
    nlabels=len(labels)
    for c in labels: 
        mu=float(dfs.loc[dfs['summary']=='mean', c]) #float is crucial, otherwise it is an object        
        df.withColumn(c, F.col(c)-F.lit(mu))
    mat=RowMatrix(df.rdd.map(list))
    sims=mat.columnSimilarities(threshold=cosineThrd) #cosine
    rsim=sims.entries.toDF().toPandas() #i,j,value
    rsim.astype({'i':int, 'j':int}, copy=False)#rsim[['i', 'j']].apply(int)    
    for ir in range(len(rsim)):
        i=rsim.iloc[ir,:]['i']
        j=rsim.iloc[ir,:]['j']        
        rsim.loc[ir, 'in']=df.columns[int(i)] #int() is necessary though I did conversion
        rsim.loc[ir, 'jn']=df.columns[int(j)]
    #rebuild the symmetrical matrix
    rmat=np.zeros((nlabels,nlabels))
    rmat[rsim['i'], rsim['j']]=rsim['value']
    rmat[rsim['j'], rsim['i']]=rsim['value']
    rmat[range(nlabels), range(nlabels)]=[1]*nlabels
    if imshow:
        from matplotlib import colors        
        plt.imshow(rmat, cmap='coolwarm', norm=colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))        
        plt.colorbar()
    if dobokeh:
        pass
    return rmat, rsim, labels

def bokeh4pcorr_v1(rmat, labels, ncolors=9, height=800, width=800, title="Pearson Correlation Coefficient Heatmap"):
    import numpy as np
    import bisect
    from itertools import chain
    from collections import OrderedDict
    import bokeh
    from bokeh.plotting import figure
    from bokeh.layouts import row    
    from bokeh.models.annotations import Label
    from bokeh.palettes import BuRd , Sunset, Turbo256
    from bokeh.models import LinearColorMapper, ColorBar
    from math import pi    
    # allows visualisation in notebook
    # from bokeh.io import output_notebook
    # from bokeh.resources import INLINE
    # output_notebook(INLINE)
    from bokeh.embed import components, file_html
    from bokeh.resources import CDN
    #https://stackoverflow.com/questions/39191653/python-bokeh-how-to-make-a-correlation-plot
    if ncolors % 2 ==0: ncolors+=1
    colors = list(reversed(Turbo256)) #list(reversed(Sunset[ncolors]))
    nlabels=len(labels)
    def get_bounds(n):
        """Gets bounds for quads with n features"""
        bottom = list(chain.from_iterable([[ii]*nlabels for ii in range(nlabels)]))
        top = list(chain.from_iterable([[ii+1]*nlabels for ii in range(nlabels)]))
        left = list(chain.from_iterable([list(range(nlabels)) for ii in range(nlabels)]))
        right = list(chain.from_iterable([list(range(1,nlabels+1)) for ii in range(nlabels)]))
        return top, bottom, left, right
    def get_colors(corrArray, colors):
        """Aligns color values from palette with the correlation coefficient values"""
        ccorr = np.arange(-1, 1, 1/(len(colors)/2))
        color = []
        for value in corrArray:
            ind = bisect.bisect_left(ccorr, value)
            color.append(colors[ind-1])
        return color
    fig=figure(width=width, height=height, x_range=(0,nlabels), y_range=(0,nlabels), title=title, toolbar_location=None, tools=['hover']) 
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None
    fig.xaxis.major_label_orientation = pi/4
    fig.yaxis.major_label_orientation = pi/4
    #Add color rect using quad
    tops, bottoms, lefts, rights = get_bounds(nlabels)
    color_list = get_colors(list(rmat.flat), colors)
    fig.quad(top=tops, bottom=bottoms, left=lefts,right=rights, line_color='white',color=color_list)
    # #add text layer
    # for i in range(nlabels):
    #     for j in range(nlabels):
    #         lbl=Label(x=i+0.5, y=j+0.5, text=f"{rmat[i,j]:.2g}")
    #         fig.add_layout(lbl)
    # Set ticks with labels
    ticks = [tick+0.5 for tick in list(range(nlabels))]
    tick_dict = OrderedDict([[tick, labels[ii]] for ii, tick in enumerate(ticks)])
    fig.xaxis.ticker = ticks
    fig.yaxis.ticker = ticks
    # Override the tick labels 
    fig.xaxis.major_label_overrides = tick_dict
    fig.yaxis.major_label_overrides = tick_dict

    # Setup color bar
    mapper = LinearColorMapper(palette=colors, low=-1, high=1)
    color_bar = ColorBar(color_mapper=mapper, location=(0, 0))
    fig.add_layout(color_bar, 'right')
    # bokeh.io.output_file(f'test/bokeh.html')
    html=file_html(fig, CDN, title)
    displayHTML(html)
    # bokeh.io.show(fig)
def bokeh4pcorr(rmat, labels,cmap='coolwarm',  width=800, height=800, title="Pearson Correlation", theme='light_minimal'):
    import numpy as np
    import bisect
    import matplotlib as mpl
    from bokeh.plotting import figure, show
    from bokeh.palettes import Viridis256, Cividis256, Turbo256, RdBu11
    # from bokeh.transform import linear_cmap
    from bokeh.models import LinearColorMapper, ColorBar
    # from bokeh.sampledata.les_mis import data
    #https://docs.bokeh.org/en/latest/docs/examples/topics/categorical/les_mis.html
    N=len(labels)
    # cmap=list(linear_cmap(field_name='pcorr', palette='Turbo256', low=-1, high=1))    
    # display(cmap)
    # colors=list(reversed(Turbo256)) #list(reversed(RdBu11)) #
    # def get_colors(rmat, colors, amin=-1, amax=1):
    def get_colors(rmat, cmap='coolwarm', amin=-1, amax=1, ncolors=256):
        """Aligns color values from palette with the correlation coefficient values"""
        # ccorr = np.arange(amin, amax, 1/ncolors)
        colorMat = ['']*np.prod(rmat.shape)
        cmap=mpl.colormaps.get_cmap(cmap)
        norm=mpl.colors.Normalize(vmin=amin, vmax=amax)
        for i,value in enumerate(rmat.flat):
            # ind = bisect.bisect_left(ccorr, value)
            # color.append(colors[ind-1])
            rgba=cmap(norm(value))
            colorMat[i]=f'{mpl.colors.to_hex(rgba)}'
        #for colorbar
        colors=['']*(ncolors)
        for i,value in enumerate(np.arange(amin, amax, 1/(ncolors/2))):   
            rgba=cmap(norm(value))            
            colors[i]=f'{mpl.colors.to_hex(rgba)}'
        return colorMat , colors
    xname = []
    yname = []
    colorMat, colors = get_colors(rmat,cmap='coolwarm', amin=-1, amax=1)
    alpha = [] # [1]*N*N
    for i, nodeR in enumerate(labels):
        for j, nodeC in enumerate(labels):
            xname.append(nodeR)
            yname.append(nodeC)
            alpha.append(min(rmat[i,j], 0.9) + 0.1 ) 
            # color.append(cmap[int(np.clip(np.floor((rmat[i,j] + 1) / 2 * (len(cmap) - 1)), 0, len(cmap) - 1))])
    #set theme before calling fig            
    from bokeh.io import curdoc
    curdoc().theme=theme #or 'caliber'    
    #prepare data source
    data=dict(xname=xname, yname=yname, colors=colorMat, alphas=alpha, pcorr=list(rmat.flat))
    fig =figure(title=title, width=width, height=height, x_axis_location='above', tools='hover,save', x_range=list(reversed(labels)), y_range=labels, tooltips=[('row', '@yname'), ('col', '@xname'), ('pcorr', '@pcorr')], background_fill_color='#ffffff', background_fill_alpha =1)
    fig.grid.grid_line_color = None
    fig.axis.axis_line_color = None
    fig.axis.major_tick_line_color = None
    # fig.axis.major_label_text_font_size = "7px"
    fig.axis.major_label_standoff = 0
    fig.xaxis.major_label_orientation = np.pi/4
    # fig.yaxis.major_label_orientation = np.pi/4
    fig.rect('xname', 'yname', 1, 1, source=data, color='colors', alpha='alphas', line_color=None, hover_line_color='black', hover_color='colors')
    # Setup color bar
    mapper = LinearColorMapper(palette=colors, low=-1, high=1)
    color_bar = ColorBar(color_mapper=mapper, location=(0, 0))
    fig.add_layout(color_bar, 'right')
    #display in Databricks
    from bokeh.embed import components, file_html
    from bokeh.resources import CDN    
    html=file_html(fig, CDN, title)
    displayHTML(html)

# rmat, rsim, labels=pcorr4sql(f""" select {",".join(lstFeatures40)} from  Summary""", imshow=True)
# # bokeh4pcorr_v1(rmat,labels, width=800,height=800, title="Pearson Correlation Coefficient Heatmap")
# bokeh4pcorr(rmat, labels, width=900, height=800, title="Pearson Correlation")
# # displayHTML('test/bokeh.html')