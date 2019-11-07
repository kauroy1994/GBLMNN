from FGBoost import MetricBoost

def construct_features(data,examples,bk):
    """construct features using some methodology
       hard coded right now..

       hope is to use counts of relational random
       walks to instantiate vectors for objects of
       classification intereset

       OUTPUT: list of feature vectors for all
       objects of classification interest in data
       in this case man index by target
    """

    #in this data-set there are men m1-m7
    #atomic features are owns dog, and st/lt relationship,
    #and compound feature is owns dog and st/lt relationship,
    #existential semantics, counts could also be used.
    data_representation = {'h(m1)': [1,1,0,1,0],
                           'h(m2)': [1,1,0,1,0],
                           'h(m3)': [1,1,0,1,0],
                           'h(m4)': [1,0,1,0,1],
                           'h(m5)': [0,1,0,0,0],
                           'h(m6)': [0,0,1,0,0],
                           'h(m7)': [0,0,1,0,0]}

    return (data_representation)

def GBLMNN(data,pos,neg,bk,target):
    """shows an example of metric learning using Rel-GBLMNN
       Relational Gradient Boosted Large Margin Nearest Neighbor
       
       this is data about men,women and dogs
       h(man) means man is happy
       o(man,dog) means man owns dog
       r(man,woman,term) means man is in relationship with woman for long term or short term
    """

    data_rep = construct_features(data,pos+neg,bk)
    r = len(data_rep[pos[0]])

    phi = []
    #learn r dimensional non linear map
    for i in range(r):

        #set boosted tree initialization (bc non convex)
        #later do LMNN initialization ..
        prior = {}
        for p in pos:
            prior[p] = data_rep[p][i]
        for n in neg:
            prior[n] = data_rep[n][i]

        #learn phi_q with boosting with 20 trees
        metric = MetricBoost(data,pos,neg,bk,prior,target)
        metric.learn(k=20)
        phi.append(metric)
        #metric_i_values = metric.infer(data,pos+neg,k=20)
                
    return (phi)
    

if __name__ == '__main__':

    data = ['o(m1,d1)','r(m1,w1,st)',
            'o(m2,d2)','r(m2,w2,st)',
            'o(m3,d3)','r(m3,w3,st)',
            'o(m4,d4)','r(m4,w4,lt)',
            'r(m5,w5,st)',
            'r(m6,w6,lt)',
            'r(m7,w7,lt)']
    
    pos = ['h(m1)',
           'h(m2)',
           'h(m4)',
           'h(m6)']

    neg = ['h(m3)',
           'h(m5)',
           'h(m7)']

    bk = ['h(+man)',
          'o(+man,-dog)',
          'r(+man,-woman,#term)']

    target = 'h'

    phi = GBLMNN(data,pos,neg,bk,target)

    #===========test example first positive example to see learned phi==
    '''
    ex = pos[0]
    phi_ex = []
    for q in phi:
        phi_ex.append(q.infer(data,[ex],k=20)[0])
    print (ex)
    print (phi_ex)
    '''
