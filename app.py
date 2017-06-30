#!/usr/bin/env python
import os
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import copy
import requests
from IPython.display import display, HTML

DEBUG = True
app = Flask(__name__)

app.secret_key = os.environ.get('APP_SECRET_KEY', 'QqWwEeRrAaSsDdFfZzXxCcVv@@!!17502')
MAILGUN_API_KEY = os.environ.get('MAILGUN_API_KEY', '')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/methods',methods = ['POST', 'GET'])
def methods():
   if request.method == 'POST':

       if request.form['method']=='method1':
           return render_template('method1.html')
       if request.form['method']=='method2':
           return render_template('method2.html')


@app.route('/result1',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':

        study = [row['study'] for row in request.json]

        g1_sample = [row['g1_sample'] for row in request.json]
        g1_mean = [row['g1_mean'] for row in request.json]
        g1_sd = [row['g1_sd'] for row in request.json]

        g2_sample = [row['g2_sample'] for row in request.json]
        g2_mean = [row['g2_mean'] for row in request.json]
        g2_sd = [row['g2_sd'] for row in request.json]

        table=[study, g1_sample, g1_mean, g1_sd, g2_sample, g2_mean, g2_sd]
        df=pd.DataFrame(table)
        df = df.transpose()

        df.index+=1

        df = df.convert_objects(convert_numeric=True)

        df['SE']=(((df[1]-1)*df[3]**2+(df[4]-1)*df[6]**2)/(df[1]+df[4]-2))**0.5

        df['d']=(df[5]-df[2])/df['SE']
        df['g']=df['d']*(1-(3/(4*(df[1]+df[4])-9)))

        df['n']=df[1]+df[4]
        df['n_1']=(1/df[1])+(1/df[4])

        df['SEd']=((df['n_1']+(df['d']**2/(2*df['n']))))**0.5
        df['SEg']=df['SEd']*(1-3/(4*(df['n'])-9))

        df['d_lower']=df['d']-1.96*df['SEd']
        df['d_upper']=df['d']+1.96*df['SEd']

        df['g_lower']=df['g']-1.96*df['SEg']
        df['g_upper']=df['g']+1.96*df['SEg']

        df['w_s_d']=1/df['SEd']**2
        df['d_s']=df['w_s_d']*df['d']

        df['w_s_g']=1/df['SEg']**2
        df['g_s']=df['w_s_g']*df['g']


        d_total=np.sum(df['d_s'])/np.sum(df['w_s_d'])
        s_total=(1/np.sum(df['w_s_d']))**0.5

        lower_d=d_total-1.96*s_total
        upper_d=d_total+1.96*s_total

        g_total=np.sum(df['g_s'])/np.sum(df['w_s_g'])
        sg_total=(1/np.sum(df['w_s_g']))**0.5

        lower_g=g_total-1.96*sg_total
        upper_g=g_total+1.96*sg_total


        q=np.sum(df['w_s_d']*df['d']**2)-((np.sum(df['d_s'])**2)/np.sum(df['w_s_d']))
        if q==0:
            print('Q=0 and I^2 is not calculable')
            I2=0
        else:
            I2=(q-8)/q

        df.drop(['n','n_1','SE'], axis=1, inplace=True)


        df.columns = ['Study', 'Group1-sample', 'Group1-mean' , 'Group1-sd' , 'Group2-sample', 'Group2-mean' , 'Group2-sd' ,'d'	,'g', 'SEd'	,'SEg','d_lower','d_upper','g_lower', 'g_upper' ,'weight (d)' ,'weighted d', 'weight (g)','weighted g']


        df2=df.to_dict(orient="dict")


        writer = pd.ExcelWriter('Meta-Mar_analysis_result.xlsx')
        df.to_excel(writer,'Sheet1')
        writer.save()


    return render_template("result1.html", total=HTML(df.to_html()), ave_d=float("{0:.2f}".format(d_total)), ave_SEd=float("{0:.2f}".format(s_total)),lower_dd=float("{0:.2f}".format(lower_d)),upper_dd=float("{0:.2f}".format(upper_d)),ave_g=float("{0:.2f}".format(g_total)), ave_SEg=float("{0:.3f}".format(sg_total)),lower_gg=float("{0:.3f}".format(lower_g)),upper_gg=float("{0:.3f}".format(upper_g)), Het=100*float("{0:.3f}".format(I2)))

@app.route('/return-file/')
def return_file():
    return send_file('Meta-Mar_analysis_result.xlsx', attachment_filename='Meta-Mar_analysis_result.xlsx')

@app.route('/method2')
def method2():
    return render_template('method2.html')


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        xl=pd.ExcelFile(f)
        df=xl.parse('Data')

        def SE(sd_c,n_c,sd_a,n_a):
            SE=(((n_c-1)*sd_c**2+(n_a-1)*sd_a**2)/(n_c+n_a-2))**0.5
            return SE

        def d(s,m_c,m_a):
            d=(m_a-m_c)/s
            return abs(d)

        def g(d,n_c,n_a):
            g=d*(1-(3/(4*(n_c+n_a)-9)))
            return g


        s_per_study=SE(df['Group1-sd'],df['Group1-sample size'],df['Group2-sd'],df['Group2-sample size'])

        d_per_study=abs(d(s_per_study,df['Group2-mean'],df['Group1-mean']))

        g_per_study=g(d_per_study, df['Group1-sample size'], df['Group2-sample size'])

        n=df['Group1-sample size']+df['Group2-sample size']
        n_1=(1/df['Group1-sample size'])+(1/df['Group2-sample size'])

        se_d_per_study=((n_1+(d_per_study**2/(2*n))))**0.5
        se_g_per_study=se_d_per_study*(1-3/(4*(n)-9))

        d_lower_per_study=d_per_study-1.96*se_d_per_study
        d_upper_per_study=d_per_study+1.96*se_d_per_study

        g_lower_per_study=g_per_study-1.96*se_g_per_study
        g_upper_per_study=g_per_study+1.96*se_g_per_study



        w_s_d=1/se_d_per_study**2
        d_s=w_s_d*d_per_study

        w_s_g=1/se_g_per_study**2
        g_s=w_s_g*g_per_study


        d_total=np.sum(d_s)/np.sum(w_s_d)
        s_total=(1/np.sum(w_s_d))**0.5

        lower_d=d_total-1.96*s_total
        upper_d=d_total+1.96*s_total

        g_total=np.sum(g_s)/np.sum(w_s_g)
        sg_total=(1/np.sum(w_s_g))**0.5

        lower_g=g_total-1.96*sg_total
        upper_g=g_total+1.96*sg_total

        df["d_lower"]=d_lower_per_study
        df["d"]=d_per_study
        df["d_upper"]=d_upper_per_study

        df["g_lower"]=g_lower_per_study
        df["g"]=g_per_study
        df["g_upper"]=g_upper_per_study


        q=np.sum(w_s_d*d_per_study**2)-((np.sum(d_s)**2)/np.sum(w_s_d))
        I2=(q-8)/q
        if q==0:
            I2=0
        else:
            I2=(q-8)/q

        df.index+=1

        writer = pd.ExcelWriter('Meta-Mar_analysis_result.xlsx')
        df.to_excel(writer,'Sheet1')
        writer.save()

        study_list=map(lambda s: str(s), df['Study'].tolist())
        d_list=df['d'].tolist()
        d_lower_list=df['d_lower'].tolist()
        d_upper_list=df['d_upper'].tolist()

        return render_template("result2.html", study_list=study_list, d_list=d_list, d_lower_list=d_lower_list , d_upper_list=d_upper_list, total=HTML(df.to_html()), ave_d=float("{0:.2f}".format(d_total)), ave_SEd=float("{0:.2f}".format(s_total)),lower_dd=float("{0:.2f}".format(lower_d)),upper_dd=float("{0:.2f}".format(upper_d)),ave_g=float("{0:.2f}".format(g_total)), ave_SEg=float("{0:.3f}".format(sg_total)),lower_gg=float("{0:.3f}".format(lower_g)),upper_gg=float("{0:.3f}".format(upper_g)),   Het=100*float("{0:.3f}".format(I2)))


@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/meta')
def meta():
    return render_template('meta.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/submitcontact', methods = ['POST', 'GET'])
def submitcontact():
    if request.method == 'POST':
        your_name = request.form['your_name']
        your_email = request.form['your_email']
        your_message = request.form['your_message']

        requests.post("https://api.mailgun.net/v3/samples.mailgun.org/messages",
            auth=("api", MAILGUN_API_KEY),
            data={
                "from": "%s <%s>" % (your_name, your_email),
                "to": ["s.ashkan.beheshti@gmail.com"],
                "subject": "Meta_Analysis Client",
                "text": your_message
            })
        return render_template("sent.html")




if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=DEBUG, host='0.0.0.0', port=port)
