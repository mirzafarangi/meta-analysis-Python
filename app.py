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
ELASTICMAIL_API_KEY = os.environ.get('ELASTICMAIL_API_KEY', '')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result1',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        try:
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

            df.columns = [
                'Study', 'Group1-sample', 'Group1-mean', 'Group1-sd', 'Group2-sample', 'Group2-mean', 'Group2-sd',
                'd', 'g', 'SEd', 'SEg', 'd_lower', 'd_upper', 'g_lower', 'g_upper',
                'weight (d)', 'weighted d', 'weight (g)', 'weighted g'
            ]

            df2=df.to_dict(orient="dict")
            writer = pd.ExcelWriter('results/Meta-Mar_analysis_result.xlsx')
            df.to_excel(writer,'Sheet1')
            writer.save()

            resultData = {
                'result_table': HTML(df.to_html(classes="responsive-table-2 rt cf")),
                'ave_d': float("{0:.2f}".format(d_total)),
                'ave_SEd': float("{0:.2f}".format(s_total)),
                'lower_dd': float("{0:.2f}".format(lower_d)),
                'upper_dd': float("{0:.2f}".format(upper_d)),
                'ave_g': float("{0:.2f}".format(g_total)),
                'ave_SEg': float("{0:.3f}".format(sg_total)),
                'lower_gg': float("{0:.3f}".format(lower_g)),
                'upper_gg': float("{0:.3f}".format(upper_g)),
                'Het': 100*float("{0:.3f}".format(I2))
            }

            content = render_template("result1.html", **resultData)
            return jsonify({
                'content': content,
                'd_list': df['d'].tolist(),
                'study_list': df['Study'].tolist(),
                'd_lower_list': df['d_lower'].tolist(),
                'd_upper_list': df['d_upper'].tolist(),
                'ave_g': float("{0:.2f}".format(g_total)),
            })

        except Exception as error:
            print(error)
            return render_template('error_content.html')

@app.route('/return-file/')
def return_file():
    return send_file('results/Meta-Mar_analysis_result.xlsx', attachment_filename='Meta-Mar_analysis_result.xlsx')

@app.route('/method2')
def method2():
    return render_template('method2.html')


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
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

            def i(n_c,n_a):
                i=(1-(3/(4*(n_c+n_a)-9)))
                return i

            s_per_study=SE(df['Sd1'],df['N1'],df['Sd2'],df['N2'])

            d_per_study=abs(d(s_per_study,df['Mean1'],df['Mean2']))

            g_per_study=g(d_per_study, df['N1'], df['N2'])

            i_per_study=i(df['N1'], df['N2'])


            n=df['N1']+df['N2']
            n_1=(1/df['N1'])+(1/df['N2'])

            se_d_per_study=((n_1+(d_per_study**2/(2*n))))**0.5
            se_g_per_study=se_d_per_study*(1-3/(4*(n)-9))

            d_lower_per_study=d_per_study-1.96*se_d_per_study
            d_upper_per_study=d_per_study+1.96*se_d_per_study

            g_lower_per_study=g_per_study-1.96*se_g_per_study
            g_upper_per_study=g_per_study+1.96*se_g_per_study

            w_s_d=1/se_d_per_study**2
            d_s=w_s_d*d_per_study

            w_s_g_fixed=1/se_g_per_study**2
            g_s_fixed=w_s_g_fixed*g_per_study
            g2_s_fixed=w_s_g_fixed*g_per_study**2


            qq=np.sum(g2_s_fixed)-((np.sum(g_s_fixed))**2/np.sum(w_s_g_fixed))
            c=np.sum(w_s_g_fixed)-((np.sum(w_s_g_fixed**2))/(np.sum(w_s_g_fixed)))
            degf= len(df.index)-2
            if qq >= degf:
                t2=(qq-degf)/c
            else:
                t2=0

            w_s_g_random= 1/((se_g_per_study**2)+t2)
            g_s_random=w_s_g_random*g_per_study
            g_total_random=np.sum(g_s_random)/np.sum(w_s_g_random)
            sg_total_random=(1/np.sum(w_s_g_random))**0.5

            lower_g_random=g_total_random-1.96*sg_total_random
            upper_g_random=g_total_random+1.96*sg_total_random

            g_s_random=w_s_g_random*g_per_study



            d_total=np.sum(d_s)/np.sum(w_s_d)
            s_total=(1/np.sum(w_s_d))**0.5

            lower_d=d_total-1.96*s_total
            upper_d=d_total+1.96*s_total

            g_total_fixed=np.sum(g_s_fixed)/np.sum(w_s_g_fixed)
            sg_total_fixed=(1/np.sum(w_s_g_fixed))**0.5

            lower_g_fixed=g_total_fixed-1.96*sg_total_fixed
            upper_g_fixed=g_total_fixed+1.96*sg_total_fixed




            df["Cohen's d"]=d_per_study
            df["CorrectionFactor"]=i_per_study

            df["Hedges'g (SMD)"]=g_per_study
            df["SEg"]=se_g_per_study

            df["95%CI-Lower"]= g_lower_per_study
            df["95%CI-Upper"]= g_upper_per_study

            df["weight(%)-fixed model"]=(w_s_g_fixed/sum(w_s_g_fixed))*100
            df["weight(%)-random model %"]=(w_s_g_random/sum(w_s_g_random))*100





            #fixed model

            q_fixed=np.sum(w_s_g_fixed*g_per_study**2)-((np.sum(g_s_fixed)**2)/np.sum(g_s_fixed))
            if q_fixed==0:
                I2_fixed=0
            else:
                degf= len(df.index)-2
                I2_fixed=(q_fixed-degf)/q_fixed

            #random model

            q_random=np.sum(w_s_g_random*g_per_study**2)-((np.sum(g_s_random)**2)/np.sum(g_s_random))
            if q_random==0:
                I2_random=0
            else:
                degf= len(df.index)-2
                I2_random=(q_random-degf)/q_random

            df2=pd.DataFrame(index=['Fixed Effect Model','Random Effect Model'], columns=["Hedges's g",'SEg', "95%CI lower", "95%CI upper", 'Heterogeneity %'])
            df2.xs('Fixed Effect Model')["Hedges's g"]= g_total_fixed
            df2.xs('Fixed Effect Model')["SEg"]= sg_total_fixed
            df2.xs('Fixed Effect Model')["95%CI lower"]= lower_g_fixed
            df2.xs('Fixed Effect Model')["95%CI upper"]= upper_g_fixed
            df2.xs('Fixed Effect Model')['Heterogeneity %']= I2_fixed
            df2.xs('Random Effect Model')["Hedges's g"]= g_total_random
            df2.xs('Random Effect Model')["SEg"]= sg_total_random
            df2.xs('Random Effect Model')["95%CI lower"]= lower_g_random
            df2.xs('Random Effect Model')["95%CI upper"]= upper_g_random
            df2.xs('Random Effect Model')['Heterogeneity %']= I2_random



            #Save the analysis

            df.index+=1

            writer = pd.ExcelWriter('results/Meta-Mar_analysis_result.xlsx')
            df.to_excel(writer,'Results per study')
            df2.to_excel(writer,'Total Results')

            writer.save()

            study_list = list(map(lambda x: str(x), df['Study'].tolist()))
            resultData = {
                'study_list': study_list,
                'g_list': df["Hedges'g (SMD)"].tolist(),
                'g_lower_list': df["95%CI-Lower"].tolist(),
                'g_upper_list': df["95%CI-Upper"].tolist(),
                'total': HTML(df.to_html()),
                'ave_g_fixed': float("{0:.2f}".format(g_total_fixed)),
                'ave_SEg_fixed': float("{0:.2f}".format(sg_total_fixed)),
                'lower_gg_fixed': float("{0:.2f}".format(lower_g_fixed)),
                'upper_gg_fixed': float("{0:.2f}".format(upper_g_fixed)),
                'ave_g_random': float("{0:.2f}".format(g_total_random)),
                'ave_SEg_random': float("{0:.3f}".format(sg_total_random)),
                'lower_gg_random': float("{0:.3f}".format(lower_g_random)),
                'upper_gg_random': float("{0:.3f}".format(upper_g_random)),
                'Het_fixed': 100 * float("{0:.3f}".format(I2_fixed)),
                'Het_random': 100 * float("{0:.3f}".format(I2_random))
            }

            return render_template("result2.html", **resultData)
        except Exception as error:
            print(error)
            return render_template('error_content.html')

@app.route('/about')
def about():
    return render_template('about.html')

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

        requests.post("https://api.elasticemail.com/v2/email/send",
            data={
                "apiKey": ELASTICMAIL_API_KEY,
                "from": your_email,
                "fromName": your_name,
                "to": ["s.ashkan.beheshti@gmail.com"],
                "subject": "Meta Mar User",
                "bodyHtml": your_message
            })
        return render_template("sent.html")


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=DEBUG, host='0.0.0.0', port=port)
