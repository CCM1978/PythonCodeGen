from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy import inf

class Learning:
    def __init__(self, xtrain, xtest):
        self.xtrain = xtrain
        self.xtest = xtest
    def SearchIndex(self):
        # TF-IDF
        pipe = Pipeline([('count', CountVectorizer(min_df= 0 , ngram_range=(0,1),vocabulary=self.xtest)),
                            ('tfid', TfidfTransformer(smooth_idf=False))]).fit(self.xtrain)
        pipe_count = pipe['count'].transform(self.xtrain).toarray()
        pipe_idf = pipe['tfid'].idf_#Hasil IDF query per kata
        a = np.array(pipe_count)
        b = np.array(pipe_idf)
        b[b == inf] = 0
        aa = a.reshape(len(a), len(b))
        ba = b.reshape(1, len(b))
        #Cosine Similarity
        cos_lib = cosine_similarity(aa, ba, dense_output=False)#Hasil score cosine similarity
        maxElement = np.amax(cos_lib)
        result = np.where(cos_lib == maxElement)
        res = result[0]#Hasil pencarian index
        #-------------------------------------------------------------------
        print("Pembobotan TF-IDF : \n")
        for i in range(len(self.xtrain)):
            print(self.xtrain[i]," =>", pipe_count[i])
        print("----------------------------------------------------------------")
        print("Hasil TF-IDF :\n")
        for i in range(len(self.xtest)):
            print(self.xtest[i]," =>", pipe_idf[i])
        print("----------------------------------------------------------------")
        print("Hasil penilaian Cosine Similarity :\n")
        for i in range(len(self.xtrain)):
            print(self.xtrain[i]," =>", cos_lib[i])
        print("----------------------------------------------------------------")
        print("Hasil parse tree :\n")
        total = len(self.xtest)
        for i in range(len(self.xtrain)):
            bla = pipe_count[i]
            count = sum(bla)
            w = count/total
            sim_value = cos_lib[i]*w
            print(self.xtrain[i]," =>", sim_value)
        print("----------------------------------------------------------------")
        return res
