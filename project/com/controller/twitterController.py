from flask import *
import os
import datetime
from project import app

@app.route('/twitter')
def twitter():
    # all_embed=[]
    user='@narendramodi'
    fname=open('Data/Twitter/{}/with_reply_type.json'.format(user[1:]))
    json_data=json.load(fname)
    fname.close()
    return render_template('twitter.html',all_embed=json_data[:3],len=len,range=range)

@app.route('/report',methods=['POST','GET'])
def report():
    # req=request.form
    reply_id=request.form.get('reply_id')
    text=request.form.get('text')
    text=text.replace('\n',' ').replace('\r',' ').replace('\t',' ')
    retrain=request.form.get('retrain')
    file_ptr=open('Data/retrain/retrain.tsv','a')
    data='{}\t{}\t{}\n'.format(reply_id,text,retrain)
    file_ptr.write(data)
    file_ptr.close()

    return 'success'