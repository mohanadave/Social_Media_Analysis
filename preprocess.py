for msg in ['hello world',
            'He is doing for his country men. Immigrants are secondary. Syria was a fuck up. But on domestic side he is doing well. Illegal crossing has decreased. Business are going to US. He is doing well.',
            'fucker bastards, bitches']:
    temp_start = datetime.datetime.now()
    test_message = preprocess(msg)
    classify(test_message, 'lr_best_multiclass_2019_07_01_14_35_59.smamodel', 'tfidf_lemma_2019_07_01_14_35_59.smavec')
    temp_end = datetime.datetime.now()
    print('time for {}:- {}'.format(msg, temp_end - temp_start))
end = datetime.datetime.now()
print('total time{}:- '.format(end - start))
