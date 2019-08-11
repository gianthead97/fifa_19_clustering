
import graphlab as gl

df = gl.SFrame('preprocessed_data.csv')
#print(df)
gl.canvas.set_target('browser')

df.show()
#model2 = gl.dbscan.create(df, radius=0.25, min_core_neighbors=3)
#print(model2.summary())

#model2.show()
