#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Akash and Brian's Final Project
# Data Set - Songs 
# Used the K-Means - tf-id and LDA
# which artists used common words and formed clusters accordingly. 


# In[3]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics 

dir_file = os.getcwd() # returns path to current directory
files_dir = os.listdir(dir_file)  # list of files in current directory

csv_files = [f for f in files_dir if f.endswith('csv')]
song_file = csv_files[0]

fid = open(song_file)
song_df = pd.read_csv(song_file)
print(song_df.columns)  # number of rows - number of songs
print(song_df.shape) # number of fields - columns


# In[5]:


from collections import Counter 

artist_dict = Counter(song_df['Artist'])  
print("Chosen Artist: ")
#choose_artist = ['Cardi B', '21 Savage', 'Drake']
choose_artist = ['Cardi B', '21 Savage', 'Drake','Kodak Black','Playboi Cari', 'Gunna','Rich The Kid', '6ix9nine', 'Trippie Redd',' Lil Yachty', 'Lil Uzi Vert', 'Young Thug', 'Travis Scott', 'Lil Pump','Migos','Future','Eminem','Jay Z','Nas','Lil Wayne','Kendrick Lamar','Kanye West','Drake','Tupac Shakur','The Notorious BIG','Andre 3000', 'J Cole','DMX','Snoop Dogg','Dr Dre','Wu-Tang Clan','Sugar Hill Gang','Kurtis Blow','Afrika Bambaata','Grandmaster Flash','Run DMC','T La Rock','Slick Rick','BDP','Public Enemy','Big Daddy Kane','Public Enemy','LL Cool J','Black Sheep','Mobb Deep','Puff Daddy','50 Cent', 'Mike Jones','Three 6 Mafia','T.I.','Jay Electronica', 'Rick Ross', 'Big Sean', 'Lil Controlla']
#choose_artist = ['Cardi B', 'Tupac Shakur', 'Eminem']
#choose_artist = ['21 Savage', 'Kodak Black', 'Travis Scott']



nr_artist = len(choose_artist)
most_common_artist = [t[0] for t in artist_dict.items() if t[0] in choose_artist]
print(most_common_artist)
print("\n")
print("Following are the average number of songs by artist: ")
print(artist_dict.most_common(30))


nr_artist = 2
index_songs_top_20 = song_df['Artist'].isin(most_common_artist)   # returns a bool index
print(index_songs_top_20.shape)
selected_artists = song_df[index_songs_top_20]
print(selected_artists.shape)


# In[6]:


# For stop words, changing everything to lowercase
# Played around with the max_df, min_df, max_features

vect = CountVectorizer(stop_words = 'english',lowercase = True, max_df = .7, min_df = 1)

counter = vect.fit_transform(selected_artists['Stemmed_Genius'])
transf  = TfidfTransformer(norm = 'l2', sublinear_tf = True) 
# TfidfTransformer takes the CountVectorizer output and computes the tf-idf
tf_idf = transf.fit_transform(counter)

print(len(vect.get_feature_names()))


# In[7]:


k_clusters = 4
model = MiniBatchKMeans(n_clusters=k_clusters, init='k-means++', max_iter=100, batch_size=5000, 
                        n_init = 10, verbose = 0)
model.fit(tf_idf)
print("\nSilhouette Coefficient: %0.3f" %metrics.silhouette_score(tf_idf, model.labels_, metric = "cosine"))


# In[8]:


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]  # sort and reverse
terms = vect.get_feature_names()

for i in range(k_clusters):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:  # print first ten terms from the cluster
        print(' %s' % terms[ind]),
    print


# In[9]:


# compute homogeneity with artist names

# get artist for the selected songs
artist = selected_artists.Artist.copy()
artist = pd.Categorical(artist)


print("Homogeneity: %0.3f"  % metrics.homogeneity_score(artist, model.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(artist, model.labels_))
print("V-measure: %0.3f"    % metrics.v_measure_score(artist, model.labels_))
print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(artist, model.labels_))


# In[10]:


# compute purity 
# choose the max in each cluster = purity per cluster
# sum purity in each cluster/# number of songs
import numpy as np

purity = np.zeros((k_clusters,1))
for c in range(k_clusters):
    # extract labels of each artist and count them
    index_cluster = model.labels_ == c;
    count_artist = Counter(artist[index_cluster])
    print(c, count_artist)
    purity[c] = max(count_artist.values())

total_purity = np.sum(purity)/len(artist)
print(total_purity)


# In[13]:


# check some data points

index_Cardi_B = selected_artists['Artist'].isin(['Cardi B'])
index_Tupac_Shakur = selected_artists['Artist'].isin(['Tupac Shakur'])
index_Eminem = selected_artists['Artist'].isin(['Eminem'])
index1 = selected_artists['Artist'].isin(['21 Savage'])
index2 = selected_artists['Artist'].isin(['Kodak Black'])
index3 = selected_artists['Artist'].isin(['Playboi Cari'])
index4 = selected_artists['Artist'].isin(['Gunna'])
index5 = selected_artists['Artist'].isin(['Rich The Kid'])
index6 = selected_artists['Artist'].isin(['6ix9nine'])
index7 = selected_artists['Artist'].isin(['Trippie Redd'])
index8 = selected_artists['Artist'].isin(['Lil Yachty'])
index9 = selected_artists['Artist'].isin(['Lil Uzi Vert'])
index10 = selected_artists['Artist'].isin(['Young Thug'])
index11 = selected_artists['Artist'].isin(['Travis Scott'])
index12 = selected_artists['Artist'].isin(['Lil Pump'])
index13 = selected_artists['Artist'].isin(['Migos'])
index14 = selected_artists['Artist'].isin(['Future'])
index15 = selected_artists['Artist'].isin(['Jay Z'])
index16 = selected_artists['Artist'].isin(['Nas'])
index17 = selected_artists['Artist'].isin(['Lil Wayne'])
index18 = selected_artists['Artist'].isin(['Kendrick Lamar'])
index19 = selected_artists['Artist'].isin(['Kanye West'])
index20 = selected_artists['Artist'].isin(['Drake'])
index21 = selected_artists['Artist'].isin(['The Notorious BIG'])
index22 = selected_artists['Artist'].isin(['Kanye West'])
index23 = selected_artists['Artist'].isin(['Andre 3000'])
index24 = selected_artists['Artist'].isin(['J Cole'])
index25 = selected_artists['Artist'].isin(['DMX'])
index26 = selected_artists['Artist'].isin(['Snoop Dogg'])
index27 = selected_artists['Artist'].isin(['Dr Dre'])
index28 = selected_artists['Artist'].isin(['Wu-Tang Clan'])
index29 = selected_artists['Artist'].isin(['Sugar Hill Gang'])
index30 = selected_artists['Artist'].isin(['Kurtis Blow'])
index31 = selected_artists['Artist'].isin(['Grandmaster Flash'])
index32 = selected_artists['Artist'].isin(['Run DMC'])
index33 = selected_artists['Artist'].isin(['T La Rock'])
index34 = selected_artists['Artist'].isin(['Slick Rick'])
index35 = selected_artists['Artist'].isin(['BDP'])
index36 = selected_artists['Artist'].isin(['Public Enemy'])
index37 = selected_artists['Artist'].isin(['Black Sheep'])
index38 = selected_artists['Artist'].isin(['Mobb Deep'])
index39 = selected_artists['Artist'].isin(['Puff Daddy'])
index40 = selected_artists['Artist'].isin(['50 Cent'])
index41 = selected_artists['Artist'].isin(['Mike Jones'])
index42 = selected_artists['Artist'].isin(['Three 6 Mafia'])
index43 = selected_artists['Artist'].isin(['T.I.'])
index44 = selected_artists['Artist'].isin(['Jay Electronica'])
index45 = selected_artists['Artist'].isin(['Rick Ross'])
index46 = selected_artists['Artist'].isin(['Big Sean'])
index47 = selected_artists['Artist'].isin(['Lil Controlla'])



#index_Cardi_B = selected_artists['Artist'].isin(['Travis Scott'])
#index_Tupac_Shakur = selected_artists['Artist'].isin(['21 Savage'])
#index_Eminem = selected_artists['Artist'].isin(['Kodak Black'])


print(sorted(Counter(model.labels_[index_Cardi_B]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))
print(sorted(Counter(model.labels_[index_Tupac_Shakur]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))
print(sorted(Counter(model.labels_[index_Eminem]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))


print(sorted(Counter(model.labels_[index_Cardi_B]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))
print(sorted(Counter(model.labels_[index_Tupac_Shakur]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))
print(sorted(Counter(model.labels_[index_Eminem]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))


print(sorted(Counter(model.labels_[index1]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))
print(sorted(Counter(model.labels_[index2]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))
print(sorted(Counter(model.labels_[index3]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))



print(sorted(Counter(model.labels_[index_Cardi_B]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))
print(sorted(Counter(model.labels_[index_Tupac_Shakur]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))
print(sorted(Counter(model.labels_[index_Eminem]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))



print(sorted(Counter(model.labels_[index_Cardi_B]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))
print(sorted(Counter(model.labels_[index_Tupac_Shakur]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))
print(sorted(Counter(model.labels_[index_Eminem]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))




print(sorted(Counter(model.labels_[index_Cardi_B]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))
print(sorted(Counter(model.labels_[index_Tupac_Shakur]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))
print(sorted(Counter(model.labels_[index_Eminem]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))

print(sorted(Counter(model.labels_[index_Cardi_B]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))
print(sorted(Counter(model.labels_[index_Tupac_Shakur]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))
print(sorted(Counter(model.labels_[index_Eminem]).items(),key = 
             lambda kv:(kv[1], kv[0]), reverse =True))



# In[14]:


model.labels_[index_Tupac_Shakur]


# In[15]:


model.labels_[index_Cardi_B]


# In[16]:


# SVD
# project Tfidf model onto singular value decomposition - LSI transform and then do clustering
from sklearn.decomposition import TruncatedSVD

dim = 15 # After analysis
svd = TruncatedSVD(n_components=dim, n_iter = 10)

lsi = svd.fit_transform(tf_idf)
print(lsi.shape)


# In[17]:


# checking singular values

explained_variance = svd.explained_variance_ratio_.sum()
print("Sum of explained variance ratio: %d%%" % (int(explained_variance * 100)))

print(svd.singular_values_[:min(dim,100)])  


# In[18]:


k_clusters = 4
model_lsi = MiniBatchKMeans(n_clusters=k_clusters, init='k-means++', max_iter=200, batch_size=5000, 
                        n_init = 10)
model_lsi.fit(lsi)
#print("\nSilhouette Coefficient: %0.3f" %metrics.silhouette_score(tf_idf, model.labels_, metric = "cosine"))


# In[19]:


# compute homogeneity with artist labels

artist = selected_artists.Artist.copy()
artist = pd.Categorical(artist)

print("Homogeneity: %0.3f"  % metrics.homogeneity_score(artist, model_lsi.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(artist, model_lsi.labels_))
print("V-measure: %0.3f"    % metrics.v_measure_score(artist, model_lsi.labels_))
print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(artist, model_lsi.labels_))


# In[20]:


# print top topic indices per cluster
print("Top top words per cluster:")
terms = vect.get_feature_names()

# find song words in each cluster, sum up their counts and print the top words
for k in range(k_clusters):
    index_songs_k = model_lsi.labels_ == k
    count_terms_k = sum(counter[index_songs_k,:])   # sum per columns 
    order_terms = count_terms_k.toarray().argsort()[:,::-1].ravel()  # convert to a 1D array
    print('Cluster', k)
    
    for t in order_terms[:10]:
        print('\t', terms[t], count_terms_k[0,t])

#order_centroids = model_lda.cluster_centers_.argsort()[:, ::-1]  # sort and reverse
#for i in range(k_clusters):
#print("Cluster %d:" % i),
#    for ind in order_centroids[i, :5]:  # print first ten terms from the cluster
#.   print(ind, model_lda.cluster_centers_[i,ind])


# In[21]:


# compute purity 

# choose the max in each cluster = purity per cluster
# sum purity in each cluster
import numpy as np

purity = np.zeros((k_clusters,1))
for c in range(k_clusters):
    index_cluster = model_lsi.labels_ == c;
    count_artist = Counter(artist[index_cluster])
    print(c, count_artist)
    purity[c] = max(count_artist.values())

total_purity = np.sum(purity)/len(artist)
print(total_purity)


# In[22]:


# plot clusters
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD
pca = TruncatedSVD(n_components = 2)
#print('explained variance')
pca.fit(tf_idf)
x2 = pca.transform(tf_idf)


# In[23]:


print(x2.shape)
# add labels
data_x2 = pd.DataFrame(x2, columns = ['x','y'])
data_x2['label'] = model.labels_
data_x2['orig_label'] = artist
data_x2.head()


# In[24]:


sb.lmplot(data=data_x2, x='x', y='y', hue='orig_label',fit_reg=False, legend=True, legend_out=True) 


# In[ ]:





# In[25]:


from collections import Counter 

# LDA with sklearn
from sklearn.decomposition import LatentDirichletAllocation

num_of_topics = 3
lda_transf = LatentDirichletAllocation(
            n_components = num_of_topics, max_iter= 10, 
            learning_method = 'online', batch_size = 1000)   # 'online' - faster, uses subset of data
lda = lda_transf.fit_transform(counter)
print(lda.shape)
print(lda_transf.components_.shape)


# In[26]:


# get distribution of each document over the 3 topics
print(lda[10,:])


# In[27]:


# get largest distribution of words over topics
feature_names = vect.get_feature_names()  # feature_names 
for topic_idx, topic in enumerate(lda_transf.components_):
        print("Topic %d:" % (topic_idx))
        words = []
        for i in topic.argsort()[:-10 - 1:-1]:
            words.append(feature_names[i])
        print(words)


# In[28]:


# get distribution of drama and comedy films over topics
import numpy as np

#artist1 =  selected_artists['Artist'].isin(['Travis Scott'])
#artist2 = selected_artists['Artist'].isin(['21 Savage'])
#artist3  = selected_artists['Artist'].isin(['Kodak Black'])

artist1 =  selected_artists['Artist'].isin(['Cardi B'])
artist2 = selected_artists['Artist'].isin(['Tupac Shakur'])
artist3  = selected_artists['Artist'].isin(['Eminem'])
index_Eminem = selected_artists['Artist'].isin(['Eminem'])
index1 = selected_artists['Artist'].isin(['21 Savage'])
index2 = selected_artists['Artist'].isin(['Kodak Black'])
index3 = selected_artists['Artist'].isin(['Playboi Cari'])
index4 = selected_artists['Artist'].isin(['Gunna'])
index5 = selected_artists['Artist'].isin(['Rich The Kid'])
index6 = selected_artists['Artist'].isin(['6ix9nine'])
index7 = selected_artists['Artist'].isin(['Trippie Redd'])
index8 = selected_artists['Artist'].isin(['Lil Yachty'])
index9 = selected_artists['Artist'].isin(['Lil Uzi Vert'])
index10 = selected_artists['Artist'].isin(['Young Thug'])
index11 = selected_artists['Artist'].isin(['Travis Scott'])
index12 = selected_artists['Artist'].isin(['Lil Pump'])
index13 = selected_artists['Artist'].isin(['Migos'])
index14 = selected_artists['Artist'].isin(['Future'])
index15 = selected_artists['Artist'].isin(['Jay Z'])
index16 = selected_artists['Artist'].isin(['Nas'])
index17 = selected_artists['Artist'].isin(['Lil Wayne'])
index18 = selected_artists['Artist'].isin(['Kendrick Lamar'])
index19 = selected_artists['Artist'].isin(['Kanye West'])
index20 = selected_artists['Artist'].isin(['Drake'])
index21 = selected_artists['Artist'].isin(['The Notorious BIG'])
index22 = selected_artists['Artist'].isin(['Kanye West'])
index23 = selected_artists['Artist'].isin(['Andre 3000'])
index24 = selected_artists['Artist'].isin(['J Cole'])
index25 = selected_artists['Artist'].isin(['DMX'])
index26 = selected_artists['Artist'].isin(['Snoop Dogg'])
index27 = selected_artists['Artist'].isin(['Dr Dre'])
index28 = selected_artists['Artist'].isin(['Wu-Tang Clan'])
index29 = selected_artists['Artist'].isin(['Sugar Hill Gang'])
index30 = selected_artists['Artist'].isin(['Kurtis Blow'])
index31 = selected_artists['Artist'].isin(['Grandmaster Flash'])
index32 = selected_artists['Artist'].isin(['Run DMC'])
index33 = selected_artists['Artist'].isin(['T La Rock'])
index34 = selected_artists['Artist'].isin(['Slick Rick'])
index35 = selected_artists['Artist'].isin(['BDP'])
index36 = selected_artists['Artist'].isin(['Public Enemy'])
index37 = selected_artists['Artist'].isin(['Black Sheep'])
index38 = selected_artists['Artist'].isin(['Mobb Deep'])
index39 = selected_artists['Artist'].isin(['Puff Daddy'])
index40 = selected_artists['Artist'].isin(['50 Cent'])
index41 = selected_artists['Artist'].isin(['Mike Jones'])
index42 = selected_artists['Artist'].isin(['Three 6 Mafia'])
index43 = selected_artists['Artist'].isin(['T.I.'])
index44 = selected_artists['Artist'].isin(['Jay Electronica'])
index45 = selected_artists['Artist'].isin(['Rick Ross'])
index46 = selected_artists['Artist'].isin(['Big Sean'])
index47 = selected_artists['Artist'].isin(['Lil Controlla'])





print(lda[artist1].mean(axis = 0))
print(lda[artist2].mean(axis = 0))
print(lda[artist3].mean(axis = 0))


# In[29]:


# cluster based on LDA model using k means
k_clusters = 4
model_lda = MiniBatchKMeans(n_clusters=k_clusters, init='k-means++', max_iter=200, batch_size=1000, 
                        n_init = 10)
model_lda.fit(lda)


# In[ ]:





# In[30]:


# compute purity 

import numpy as np

#print(artist.shape, index_cluster1.shape, index_cluster2.shape)
purity = np.zeros((k_clusters,1))
for c in range(k_clusters):
    # extract labels
    index_cluster = model_lda.labels_ == c;
    count_artist = Counter(artist[index_cluster])
    print(c, count_artist)
    purity[c] = max(count_artist.values())

total_purity = np.sum(purity)/len(artist)
print(total_purity)


# In[126]:


# print top topic indices per cluster
print("Top topic indices per cluster:")
order_centroids = model_lda.cluster_centers_.argsort()[:, ::-1]  # sort and reverse

for i in range(k_clusters):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :5]:  # print first terms from the cluster
        print(ind, model_lda.cluster_centers_[i,ind])
  


# In[134]:


# find the closest artist to a another artist using tf-idf, lda and lsi embeddings

def similar_artists(artist_id, all_embed):
    nr_artists   = all_embed.shape[0]  # number of rows 
    embed = all_embed[artist_id,:]
    
    
    dist = all_embed.dot(embed.transpose())
    dist[artist_id] = 0;
    
    return dist.argmax()

def print_artists(artist_id, all_artists):
    index_artist = selected_artists.columns.get_loc('Artist')
    index_songs  = selected_artists.columns.get_loc('Stemmed_Genius')
    print(all_artists.iloc[artist_id, index_artist],'\n')
    print('\t', all_artists.iloc[artist_id,index_songs])
    
artist_id = 3
print('Original Song: ')
print_artists(artist_id, selected_artists)

print('\ntf-idf most similar')
similar_tf_idf = similar_artists(artist_id,tf_idf)
print_artists(similar_tf_idf, selected_artists)

similar_lsi   = similar_artists(artist_id,lsi)
print('\nlsi most similar')
print_artists(similar_lsi, selected_artists)


similar_lda    = similar_artists(artist_id, lda)
print('\nlda most similar')
print_artists(similar_lda, selected_artists)


# In[ ]:





# In[ ]:





# In[ ]:




