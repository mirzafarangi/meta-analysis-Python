#!/usr/bin/env python
import os
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import copy
import requests
from IPython.display import display, HTML
import statsmodels.api as sm
from sklearn import datasets
import scipy.stats

DEBUG = True
app = Flask(__name__)

app.secret_key = os.environ.get('APP_SECRET_KEY', 'QqWwEeRrAaSsDdFfZzXxCcVv@@!!17502')
ELASTICMAIL_API_KEY = os.environ.get('ELASTICMAIL_API_KEY', '')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sitemap')
def sitemap():
    return render_template('sitemap.xml')

@app.route('/effect')
def effect():
    return render_template('effect.html')

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

            try:
                moderator = [row['moderator'] for row in request.json]
            except:
                moderator=['moderator']



            table=[study, g1_sample, g1_mean, g1_sd, g2_sample, g2_mean, g2_sd, moderator]
            df=pd.DataFrame(table)
            df = df.transpose()

            df.index+=1

            df = df.convert_objects(convert_numeric=True)

            df['Spooled']=(((df[1]-1)*df[3]**2+(df[4]-1)*df[6]**2)/(df[1]+df[4]-2))**0.5

            df['d']=(df[5]-df[2])/df['Spooled']
            df['g']=df['d']*(1-(3/(4*(df[1]+df[4])-9)))

            df['n']=df[1]+df[4]
            df['n_1']=(1/df[1])+(1/df[4])

            df['SEd']=((df['n_1']+(df['d']**2/(2*df['n']))))**0.5
            df['SEg']=df['SEd']*(1-3/(4*(df['n'])-9))

            df['d_lower']=df['d']-1.96*df['SEd']
            df['d_upper']=df['d']+1.96*df['SEd']

            df['g_lower']=df['g']-1.96*df['SEg']
            df['g_upper']=df['g']+1.96*df['SEg']

            df['w_d']=1/df['SEd']**2
            df['wd']=df['w_d']*df['d']

            df['w_fixed']=1/df['SEg']**2
            df['wg_fixed']=df['w_fixed']*df['g']
            df['wg2_fixed']=df['w_fixed']*df['g']**2

            g_total_fixed=np.sum(df['wg_fixed'])/np.sum(df['w_fixed'])
            se_total_fixed=(1/np.sum(df['w_fixed']))**0.5

            lower_g_fixed=g_total_fixed-1.96*se_total_fixed
            upper_g_fixed=g_total_fixed+1.96*se_total_fixed



            qq=np.sum(df['wg2_fixed'])-((np.sum(df['wg_fixed']))**2/np.sum(df['w_fixed']))
            c=np.sum(df['w_fixed'])-((np.sum(df['w_fixed']**2))/(np.sum(df['w_fixed'])))
            degf= len(df.index)-1
            if qq<=degf or degf==0:
                t2=0
                I2_fixed=0
            else:
                t2=(qq-degf)/c
                q_fixed=qq
                I2_fixed=(q_fixed-degf)/q_fixed

            df['w_random']= 1/((df['SEg']**2)+t2)
            df['wg_random']=df['w_random']*df['g']
            df['wg2_random']=df['w_random']**df['g']**2

            g_total_random=np.sum(df['wg_random'])/np.sum(df['w_random'])
            se_total_random=(1/np.sum(df['w_random']))**0.5

            lower_g_random=g_total_random-1.96*se_total_random
            upper_g_random=g_total_random+1.96*se_total_random

            df["weight(%)-fixed model"]=(df['w_fixed']/sum(df['w_fixed']))*100
            df["weight(%)-random model %"]=(df['w_random']/sum(df['w_random']))*100

            z_score_fixed=g_total_fixed/se_total_fixed
            p_value_fixed = scipy.stats.norm.sf(abs(z_score_fixed))*2

            z_score_random=g_total_random/se_total_random
            p_value_random = scipy.stats.norm.sf(abs(z_score_random))*2

            #Heterogeneity
            #random model

            I2_random=I2_fixed

            #moderator-regression analysis

            moderator_=df[7]
            effect_size=df['g']
            moderator_ = sm.add_constant(moderator_)
            model = sm.OLS(effect_size, moderator_).fit()
            predictions = model.predict(moderator_)
            results=model.summary()
            moder=results.as_html()


            df.drop(['Spooled','d','n','n_1','SEd','d_lower','d_upper','w_d','wd','w_fixed','wg_fixed', 'wg2_fixed', 'w_random', 'wg_random', 'wg2_random'] ,inplace=True, axis=1)


            df.columns = ['Study name', 'n1', 'Mean1' , 'SD1', 'n2', 'Mean2' , 'SD2', 'Moderator Variable', 'g', 'SEg', 'g_lower', 'g_upper', 'weight(%)-fixed model', 'weight(%)-random model' ]

            df2=df.to_dict(orient="dict")
            writer = pd.ExcelWriter('results/MetaMar_result_smd.xlsx')
            df.to_excel(writer,'Sheet1')
            writer.save()

            resultData = {
                'result_table': HTML(df.to_html(classes="responsive-table-2 rt cf")),
                'ave_g_fixed': float("{0:.2f}".format(g_total_fixed)),
                'ave_SEg_fixed': float("{0:.3f}".format(se_total_fixed)),
                'lower_gg_fixed': float("{0:.3f}".format(lower_g_fixed)),
                'upper_gg_fixed': float("{0:.3f}".format(upper_g_fixed)),
                'Het_fixed': 100*float("{0:.3f}".format(I2_fixed)),
                'ave_g_random': float("{0:.2f}".format(g_total_random)),
                'ave_SEg_random': float("{0:.3f}".format(se_total_random)),
                'lower_gg_random': float("{0:.3f}".format(lower_g_random)),
                'upper_gg_random': float("{0:.3f}".format(upper_g_random)),
                'Het_random': 100*float("{0:.3f}".format(I2_random)),
                't2': float("{0:.3f}".format(t2)),
                'p_value_fixed': float("{0:.6f}".format(p_value_fixed)),
                'p_value_random': float("{0:.6f}".format(p_value_random)),
                'z_score_fixed': float("{0:.3f}".format(z_score_fixed)),
                'z_score_random': float("{0:.3f}".format(z_score_random)),
                'moder': moder
            }

            content = render_template("result1.html", **resultData)
            return jsonify({
                'content': content,
                'g_list': df['g'].tolist(),
                'study_list': df['Study name'].tolist(),
                'g_lower_list': df['g_lower'].tolist(),
                'g_upper_list': df['g_upper'].tolist(),
                'g_weight_list': df['weight(%)-random model'].tolist(),
                'ave_g': float("{0:.2f}".format(g_total_random)),
                'lower_g_ave': float("{0:.2f}".format(lower_g_random)),
                'upper_g_ave': float("{0:.2f}".format(upper_g_random))
            })

        except Exception as error:
            print(error)
            return render_template('error_content.html')

@app.route('/return-file-smd/')
def return_file_smd():
    return send_file('results/MetaMar_result_smd.xlsx', attachment_filename='MetaMar_result_smd.xlsx')


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
            degf= len(df.index)-1
            if qq<=degf or degf==0:
                t2=0
                I2_fixed=0
            else:
                t2=(qq-degf)/c
                q_fixed=qq
                I2_fixed=(q_fixed-degf)/q_fixed

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

            z_score_fixed=g_total_fixed/sg_total_fixed
            p_value_fixed = scipy.stats.norm.sf(abs(z_score_fixed))*2

            z_score_random=g_total_random/sg_total_random
            p_value_random = scipy.stats.norm.sf(abs(z_score_random))*2

            #Heterogeneity
            #random model

            I2_random=I2_fixed

            #moderator-regression analysis
            moderator_=df['Moderator']
            effect_size=g_per_study
            moderator_ = sm.add_constant(moderator_)
            model = sm.OLS(effect_size, moderator_).fit()
            predictions = model.predict(moderator_)
            results=model.summary()
            moder=results.as_html()



            df2=pd.DataFrame(index=['Fixed Effect Model','Random Effect Model'], columns=["Hedges's g",'SEg', "95%CI lower", "95%CI upper", 'Heterogeneity %'])
            df2.xs('Fixed Effect Model')["Hedges's g"]= g_total_fixed
            df2.xs('Fixed Effect Model')["SEg"]= sg_total_fixed
            df2.xs('Fixed Effect Model')["95%CI lower"]= lower_g_fixed
            df2.xs('Fixed Effect Model')["95%CI upper"]= upper_g_fixed
            df2.xs('Fixed Effect Model')['Heterogeneity %']= 100*I2_fixed
            df2.xs('Random Effect Model')["Hedges's g"]= g_total_random
            df2.xs('Random Effect Model')["SEg"]= sg_total_random
            df2.xs('Random Effect Model')["95%CI lower"]= lower_g_random
            df2.xs('Random Effect Model')["95%CI upper"]= upper_g_random
            df2.xs('Random Effect Model')['Heterogeneity %']= 100*I2_random

            #Save the analysis

            df.index+=1
            writer = pd.ExcelWriter('results/MetaMar_result_smdxl.xlsx')
            df.to_excel(writer,'Results per study')
            df2.to_excel(writer,'Total Results')
            writer.save()



            study_list = list(map(lambda x: str(x), df['Study'].tolist()))
            resultData = {
                'study_list': study_list,
                'g_list': df["Hedges'g (SMD)"].tolist(),
                'g_lower_list': df["95%CI-Lower"].tolist(),
                'g_upper_list': df["95%CI-Upper"].tolist(),
                'g_weight_list': df["weight(%)-random model %"].tolist(),
                'total': HTML(df.to_html(classes="responsive-table-2 rt cf")),
                'ave_g_fixed': float("{0:.2f}".format(g_total_fixed)),
                'ave_SEg_fixed': float("{0:.2f}".format(sg_total_fixed)),
                'lower_gg_fixed': float("{0:.2f}".format(lower_g_fixed)),
                'upper_gg_fixed': float("{0:.2f}".format(upper_g_fixed)),
                'ave_g_random': float("{0:.2f}".format(g_total_random)),
                'ave_SEg_random': float("{0:.3f}".format(sg_total_random)),
                'lower_gg_random': float("{0:.3f}".format(lower_g_random)),
                'upper_gg_random': float("{0:.3f}".format(upper_g_random)),
                'Het_fixed': 100 * float("{0:.3f}".format(I2_fixed)),
                'Het_random': 100 * float("{0:.2f}".format(I2_random)),
                't2': float("{0:.2f}".format(t2)),
                'p_value_fixed': float("{0:.6f}".format(p_value_fixed)),
                'p_value_random': float("{0:.6f}".format(p_value_random)),
                'z_score_fixed': float("{0:.3f}".format(z_score_fixed)),
                'z_score_random': float("{0:.3f}".format(z_score_random)),

                'moder': moder
            }

            return render_template("result2.html", **resultData)
        except Exception as error:
            print(error)
            return render_template('error_content.html')

@app.route('/return-file-smdxl/')
def return_file_smdxl():
    return send_file('results/MetaMar_result_smdxl.xlsx', attachment_filename='MetaMar_result_smdxl.xlsx')

@app.route('/result_corr',methods = ['POST', 'GET'])
def result_corr():
    if request.method == 'POST':
        try:
            study = [row['study'] for row in request.json]
            correlation = [row['correlation'] for row in request.json]
            sample = [row['sample'] for row in request.json]
            moderator = [row['moderator'] for row in request.json]


            table=[study, correlation, sample, moderator]
            df=pd.DataFrame(table)
            df = df.transpose()


            df = df.convert_objects(convert_numeric=True)

            #fixed

            df['Var']=1/(df[2]-3)
            df['SE']=df['Var']**0.5
            df['weight']=1/df['Var']


            df['r_lower']=df[1]-1.96*df['SE']
            df['r_upper']=df[1]+1.96*df['SE']

            df['Fisher z']=0.5*np.log((1+df[1])/(1-df[1]))
            df['z*w']=df['Fisher z']*df['weight']

            df['d']=2*df[1]/((1-df[1]**2)**0.5)
            df['d*w']=df['d']*df['weight']




            z_total=np.sum(df['z*w'])/np.sum(df['weight'])
            s_total=(1/np.sum(df['weight']))**0.5
            r_total= (-1+np.exp(2*z_total))/(1+np.exp(2*z_total))
            d_total=2*r_total/((1-r_total**2)**0.5)

            lower_r=r_total-1.96*s_total
            upper_r=r_total+1.96*s_total

            df['r*w']=df[1]*df['weight']

            z_score = d_total/s_total
            p_value = scipy.stats.norm.sf(abs(z_score))*2


            #random

            q=np.sum(df['weight']*df['d']**2)-((np.sum(df['d*w'])**2)/np.sum(df['weight']))
            c=np.sum(df['weight'])-((np.sum(df['weight']**2))/(np.sum(df['weight'])))
            degf= len(df.index)-1
            if q<=degf or degf==0:
                t2=0
                I2=0
            else:
                t2=(q-degf)/c
                q_fixed=q
                I2=(q_fixed-degf)/q_fixed

            df['w_random']=1/(df['Var']+t2)
            df['Weight(%)_fixed']=100*df['weight']/np.sum(df['weight'])
            df['Weight(%)_random']=100*df['w_random']/np.sum(df['w_random'])

            df['wz_random']=df['Fisher z']*df['w_random']
            df['wr_random']=df[1]*df['w_random']
            df['wd_random']=df['d']*df['w_random']

            z_total_random=np.sum(df['wz_random'])/np.sum(df['w_random'])
            s_total_random=(1/np.sum(df['w_random']))**0.5
            r_total_random= (-1+np.exp(2*z_total))/(1+np.exp(2*z_total))
            d_total_random=2*r_total/((1-r_total**2)**0.5)

            lower_r_random=r_total_random-1.96*s_total_random
            upper_r_random=r_total_random+1.96*s_total_random

            z_score_random = d_total_random/s_total_random
            p_value_random = scipy.stats.norm.sf(abs(z_score_random))*2



            moderator_=df[3]
            effect_size=df[1]
            moderator_ = sm.add_constant(moderator_)
            model = sm.OLS(effect_size, moderator_).fit()
            predictions = model.predict(moderator_)
            results=model.summary()
            moder=results.as_html()



            df.drop(['Var','weight','z*w','d','d*w','r*w', 'w_random', 'wz_random', 'wd_random', 'wr_random'] ,inplace=True, axis=1)


            df.columns = ['study name', 'r', 'n', 'moderator', 'SE', 'r_lower', 'r_upper', 'Fisher z', 'Weight(%)_fixed', 'Weight(%)_random']


            df.index+=1

            df2=df.to_dict(orient="dict")
            writer = pd.ExcelWriter('results/MetaMar_result_corr.xlsx')
            df.to_excel(writer,'Sheet1')
            writer.save()


            resultData = {
                'result_table': HTML(df.to_html(classes="responsive-table-2 rt cf")),


                'ave_z': float("{0:.2f}".format(z_total)),
                'ave_r': float("{0:.2f}".format(r_total)),
                'ave_SE': float("{0:.3f}".format(s_total)),
                'lower_r': float("{0:.3f}".format(lower_r)),
                'upper_r': float("{0:.3f}".format(upper_r)),
                'Het': 100*float("{0:.3f}".format(I2)),
                'p_value': float("{0:.4f}".format(p_value)),
                'z_score': float("{0:.3f}".format(z_score)),
                'ave_z_random': float("{0:.2f}".format(z_total_random)),
                'ave_r_random': float("{0:.2f}".format(r_total_random)),
                'ave_SE_random': float("{0:.3f}".format(s_total_random)),
                'lower_r_random': float("{0:.3f}".format(lower_r_random)),
                'upper_r_random': float("{0:.3f}".format(upper_r_random)),
                'Het': 100*float("{0:.3f}".format(I2)),
                't2': float("{0:.3f}".format(t2)),
                'p_value_random': float("{0:.4f}".format(p_value_random)),
                'z_score_random': float("{0:.3f}".format(z_score_random)),
                'moder': moder
            }

            content = render_template("result_corr.html", **resultData)
            return jsonify({
                'content': content,
                'r_list': df['r'].tolist(),
                'study_list': df['study name'].tolist(),
                'r_lower_list': df['r_lower'].tolist(),
                'r_upper_list': df['r_upper'].tolist(),
                'ave_r': float("{0:.2f}".format(r_total)),
                'lower_r_ave': float("{0:.2f}".format(lower_r)),
                'upper_r_ave': float("{0:.2f}".format(upper_r))
            })

        except Exception as error:
            print(error)
            return render_template('error_content.html')

@app.route('/return-file-corr/')
def return_file_corr():
    return send_file('results/MetaMar_result_corr.xlsx', attachment_filename='MetaMar_result_corr.xlsx')

@app.route('/uploader_corr', methods = ['GET', 'POST'])
def upload_file_corr():
    if request.method == 'POST':
        try:
            f = request.files['file']
            xl=pd.ExcelFile(f)
            df=xl.parse('Data')

            df.columns = ['Study','N', 'r', 'Moderator']

            #fixed

            df['Var']=1/(df['N']-3)
            df['SE']=df['Var']**0.5
            df['weight']=1/df['Var']

            df['r_lower']=df['r']-1.96*df['SE']
            df['r_upper']=df['r']+1.96*df['SE']

            df["Fisher z"]=0.5*(np.log((1+df['r'])/(1-df['r'])))

            df['z*w']=df['Fisher z']*df['weight']

            z_total=np.sum(df['z*w'])/np.sum(df['weight'])
            s_total=(1/np.sum(df['weight']))**0.5
            r_total= (-1+np.exp(2*z_total))/(1+np.exp(2*z_total))
            d_total= 2*r_total/((1-r_total**2)**0.5)

            lower_r=r_total-1.96*s_total
            upper_r=r_total+1.96*s_total

            df['d']=2*df['r']/(1-df['r']**2)
            df['d*w']=df['d']*df['weight']

            z_score = d_total/s_total
            p_value = scipy.stats.norm.sf(abs(z_score))*2

            #random

            q=np.sum(df['weight']*df['d']**2)-((np.sum(df['d*w'])**2)/np.sum(df['weight']))
            c=np.sum(df['weight'])-((np.sum(df['weight']**2))/(np.sum(df['weight'])))
            degf= len(df.index)-1
            if q<=degf or degf==0:
                t2=0
                I2=0
            else:
                t2=(q-degf)/c
                q_fixed=q
                I2=(q_fixed-degf)/q_fixed

            df['w_random']=1/(df['Var']+t2)
            df['Weight(%)_fixed']=100*df['weight']/np.sum(df['weight'])
            df['Weight(%)_random']=100*df['w_random']/np.sum(df['w_random'])


            df['wz_random']=df['Fisher z']*df['w_random']

            z_total_random=np.sum(df['wz_random'])/np.sum(df['w_random'])
            s_total_random=(1/np.sum(df['w_random']))**0.5
            r_total_random= (-1+np.exp(2*z_total_random))/(1+np.exp(2*z_total_random))
            d_total_random= 2*r_total_random/((1-r_total_random**2)**0.5)

            lower_r_random=r_total_random-1.96*s_total_random
            upper_r_random=r_total_random+1.96*s_total_random

            z_score_random = d_total_random/s_total_random
            p_value_random = scipy.stats.norm.sf(abs(z_score_random))*2

            #moderator-regression analysis
            moderator_=df['Moderator']
            effect_size=df['r']
            moderator_ = sm.add_constant(moderator_)
            model = sm.OLS(effect_size, moderator_).fit()
            predictions = model.predict(moderator_)
            results=model.summary()
            moder=results.as_html()

            df.drop(['Var','weight','z*w','d','d*w', 'w_random', 'wz_random'] ,inplace=True, axis=1)

            df.columns = ['study name', 'N', 'r', 'moderator', 'SE', 'r_lower', 'r_upper', 'Fisher z', 'Weight(%)_fixed','Weight(%)_random']

            df2=pd.DataFrame(index=['Fixed Effect Model','Random Effect Model'], columns=["Fisher z",'SE', 'r', "95%CI lower", "95%CI upper", 'Heterogeneity %'])
            df2.xs('Fixed Effect Model')["Fisher z"]= z_total
            df2.xs('Fixed Effect Model')["SE"]= s_total
            df2.xs('Fixed Effect Model')["r"]= r_total
            df2.xs('Fixed Effect Model')["95%CI lower"]= lower_r
            df2.xs('Fixed Effect Model')["95%CI upper"]= upper_r
            df2.xs('Fixed Effect Model')['Heterogeneity %']= 100*I2
            df2.xs('Random Effect Model')["Fisher z"]= z_total_random
            df2.xs('Random Effect Model')["SE"]= s_total_random
            df2.xs('Random Effect Model')["r"]= r_total_random
            df2.xs('Random Effect Model')["95%CI lower"]= lower_r_random
            df2.xs('Random Effect Model')["95%CI upper"]= upper_r_random
            df2.xs('Random Effect Model')['Heterogeneity %']= 100*I2

            #Save the analysis

            df.index+=1

            writer = pd.ExcelWriter('results/MetaMar_result_corrxl.xlsx')
            df.to_excel(writer,'per study')
            df2.to_excel(writer,'Total Results')

            writer.save()

            study_list = list(map(lambda x: str(x), df['study name'].tolist()))
            resultData = {
                'result_table': HTML(df.to_html(classes="responsive-table-2 rt cf")),
                'study_list': study_list,
                'r_list': df['r'].tolist(),
                'r_lower_list': df['r_lower'].tolist(),
                'r_upper_list': df['r_upper'].tolist(),
                'r_weight_list': df['Weight(%)_random'].tolist(),
                'ave_z': float("{0:.2f}".format(z_total)),
                'ave_r': float("{0:.2f}".format(r_total)),
                'ave_SE': float("{0:.2f}".format(s_total)),
                'lower_r': float("{0:.2f}".format(lower_r)),
                'upper_r': float("{0:.2f}".format(upper_r)),
                'Het': 100 * float("{0:.3f}".format(I2)),
                'p_value': float("{0:.6f}".format(p_value)),
                'z_score': float("{0:.3f}".format(z_score)),
                'ave_z_random': float("{0:.2f}".format(z_total_random)),
                'ave_r_random': float("{0:.2f}".format(r_total_random)),
                'ave_SE_random': float("{0:.2f}".format(s_total_random)),
                'lower_r_random': float("{0:.2f}".format(lower_r_random)),
                'upper_r_random': float("{0:.2f}".format(upper_r_random)),
                'Het': 100 * float("{0:.3f}".format(I2)),
                't2': float("{0:.3f}".format(t2)),
                'p_value_random': float("{0:.6f}".format(p_value_random)),
                'z_score_random': float("{0:.3f}".format(z_score_random)),
                'moder': moder
            }

            return render_template("result_corrxls.html", **resultData)
        except Exception as error:
            print(error)
            return render_template('error_content.html')

@app.route('/return-file-corrxl/')
def return_file_corrxl():
    return send_file('results/MetaMar_result_corrxl.xlsx', attachment_filename='MetaMar_result_corrxl.xlsx')

@app.route('/result_ratios',methods = ['POST', 'GET'])
def result_ratios():
    if request.method == 'POST':
        try:
            study = [row['study'] for row in request.json]
            g1_e = [row['g1_e'] for row in request.json]
            g1_ne = [row['g1_ne'] for row in request.json]
            g2_e = [row['g2_e'] for row in request.json]
            g2_ne = [row['g2_ne'] for row in request.json]
            moderator = [row['moderator'] for row in request.json]

            table=[study, g1_e, g1_ne, g2_e, g2_ne, moderator]
            df=pd.DataFrame(table)
            df = df.transpose()
            df = df.convert_objects(convert_numeric=True)

            df['RiskRatio']=(df[1]/(df[1]+df[2]))/(df[3]/(df[3]+df[4]))
            df['LnRR']=np.log(df['RiskRatio'])
            df['V']=(1/df[1])-(1/(df[1]+df[2]))+(1/df[3])-(1/(df[3]+df[4]))
            df['SE']=df['V']**0.5
            df['lower_lnRR']=df['LnRR']-1.96*df['SE']
            df['upper_lnRR']=df['LnRR']+1.96*df['SE']
            df['lower_RR']=np.exp(df['lower_lnRR'])
            df['upper_RR']=np.exp(df['upper_lnRR'])
            df['weight_fixed model']=1/df['V']
            df['LnRR*w_fixed']=df['weight_fixed model']*df['LnRR']
            df['LnRR2*w_fixed']=df['weight_fixed model']*df['LnRR']**2

            qq=np.sum(df['LnRR2*w_fixed'])-(np.sum(df['LnRR*w_fixed']**2))/np.sum(df['weight_fixed model'])
            c=np.sum(df['weight_fixed model'])-((np.sum(df['weight_fixed model']**2))/(np.sum(df['weight_fixed model'])))
            degf= len(df.index)-1
            if qq <= degf or degf==0:
                t2=0
                I2=0
            else:
                t2=(qq-degf)/c
                I2=(qq-degf)/qq

            df["weight_random model"]= 1/((df['V']**2)+t2)
            df['LnRR*w_random']=df['weight_random model']*df['LnRR']
            df['weight(%)_random model']=100*df['weight_random model']/np.sum(df['weight_random model'])
            df['weight(%)_fixed model']=100*df['weight_fixed model']/np.sum(df['weight_fixed model'])

            LnRR_total_random=np.sum(df['LnRR*w_random'])/np.sum(df["weight_random model"])
            se_total_random=(1/np.sum(df["weight_random model"]))**0.5
            lower_LnRR_random=LnRR_total_random-1.96*se_total_random
            upper_LnRR_random=LnRR_total_random+1.96*se_total_random
            LnRR_total_fixed=np.sum(df['LnRR*w_fixed'])/np.sum(df["weight_fixed model"])
            se_total_fixed=(1/np.sum(df["weight_fixed model"]))**0.5
            lower_LnRR_fixed=LnRR_total_fixed-1.96*se_total_fixed
            upper_LnRR_fixed=LnRR_total_fixed+1.96*se_total_fixed
            RRave_random=np.exp(LnRR_total_random)
            lower_RRave_random=np.exp(lower_LnRR_random)
            upper_RRave_random=np.exp(upper_LnRR_random)
            RRave_fixed=np.exp(LnRR_total_fixed)
            lower_RRave_fixed=np.exp(lower_LnRR_fixed)
            upper_RRave_fixed=np.exp(upper_LnRR_fixed)
            z_score_fixed=LnRR_total_fixed/se_total_fixed
            p_value_fixed = scipy.stats.norm.sf(abs(z_score_fixed))*2
            z_score_random=LnRR_total_random/se_total_random
            p_value_random = scipy.stats.norm.sf(abs(z_score_random))*2

            #moderator-regression analysis
            moderator_=df[5]
            effect_size=df['LnRR']
            moderator_ = sm.add_constant(moderator_)
            model = sm.OLS(effect_size, moderator_).fit()
            predictions = model.predict(moderator_)
            results=model.summary()
            moder=results.as_html()

            df.drop(['LnRR*w_fixed','LnRR2*w_fixed','weight_random model','LnRR*w_random','weight_fixed model'] ,inplace=True, axis=1)
            df.columns = ['Study name', 'Events-g1', 'Non-Events_g1' , 'Events-g2', 'Non-Events_g2' ,'Moderator', 'RiskRatio', 'LnRR' , 'V', 'SE', 'lower_lnRR', 'upper_lnRR', 'lower_RR', 'upper_RR', 'weight(%)_random model', 'weight(%)_fixed model' ]
            df.index+=1

            df2=df.to_dict(orient="dict")
            writer = pd.ExcelWriter('results/MetaMar_result_ratio.xlsx')
            df.to_excel(writer,'Sheet1')
            writer.save()

            resultData = {
                'result_table': HTML(df.to_html(classes="responsive-table-2 rt cf")),
                'LnRR_total_random': float("{0:.2f}".format(LnRR_total_random)),
                'RRave_random': float("{0:.2f}".format(RRave_random)),
                'se_total_random': float("{0:.3f}".format(se_total_random)),
                'lower_RRave_random': float("{0:.3f}".format(lower_RRave_random)),
                'upper_RRave_random': float("{0:.3f}".format(upper_RRave_random)),
                'Het_random': 100*float("{0:.3f}".format(I2)),
                't2':float("{0:.3f}".format(t2)),
                'LnRR_total_fixed': float("{0:.2f}".format(LnRR_total_fixed)),
                'RRave_fixed': float("{0:.2f}".format(RRave_fixed)),
                'se_total_fixed': float("{0:.3f}".format(se_total_fixed)),
                'lower_RRave_fixed': float("{0:.3f}".format(lower_RRave_fixed)),
                'upper_RRave_fixed': float("{0:.3f}".format(upper_RRave_fixed)),
                'Het_fixed': 100*float("{0:.3f}".format(I2)),
                'p_value_fixed': float("{0:.6f}".format(p_value_fixed)),
                'p_value_random': float("{0:.6f}".format(p_value_random)),
                'z_score_fixed': float("{0:.3f}".format(z_score_fixed)),
                'z_score_random': float("{0:.3f}".format(z_score_random)),
                'moder': moder
            }

            content = render_template("result_ratios.html", **resultData)
            return jsonify({
                'content': content,
                'RR_list': df['RiskRatio'].tolist(),
                'study_list': df['Study name'].tolist(),
                'lower_RR_list': df['lower_RR'].tolist(),
                'upper_RR_list': df['upper_RR'].tolist(),
                'RRave_random': float("{0:.2f}".format(RRave_random)),
                'lower_RRave_random': float("{0:.2f}".format(lower_RRave_random)),
                'upper_RRave_random': float("{0:.2f}".format(upper_RRave_random))
            })

        except Exception as error:
            print(error)
            return render_template('error_content.html')

@app.route('/return-file-ratio/')
def return_file_ratio():
    return send_file('results/MetaMar_result_ratio.xlsx', attachment_filename='MetaMar_result_ratio.xlsx')

@app.route('/uploader_ratios', methods = ['GET', 'POST'])
def uploader_ratios():
    if request.method == 'POST':
        try:
            f = request.files['file']
            xl=pd.ExcelFile(f)
            df=xl.parse('Data')

            df.columns = ['Study name','Events (g1)', 'Non-Events (g1)', 'Events (g2)', 'Non-Events (g2)', 'Moderator']
            df['RiskRatio']=(df['Events (g1)']/(df['Events (g1)']+df['Non-Events (g1)']))/(df['Events (g2)']/(df['Events (g2)']+df['Non-Events (g2)']))
            df['LnRR']=np.log(df['RiskRatio'])
            df['V']=(1/df['Events (g1)'])-(1/(df['Events (g1)']+df['Non-Events (g1)']))+(1/df['Events (g2)'])-(1/(df['Events (g2)']+df['Non-Events (g2)']))
            df['SE']=df['V']**0.5
            df['lower_lnRR']=df['LnRR']-1.96*df['SE']
            df['upper_lnRR']=df['LnRR']+1.96*df['SE']
            df['lower_RR']=np.exp(df['lower_lnRR'])
            df['upper_RR']=np.exp(df['upper_lnRR'])
            df['weight_fixed model']=1/df['V']
            df['LnRR*w_fixed']=df['weight_fixed model']*df['RiskRatio']
            df['LnRR2*w_fixed']=df['weight_fixed model']*df['RiskRatio']**2
            df['d']=df['LnRR']*(3**0.5)*3.1415

            qq=np.sum(df['weight_fixed model']*df['d']**2)-((np.sum(df['weight_fixed model']*df['d']))**2/np.sum(df['weight_fixed model']))
            c=np.sum(df['weight_fixed model'])-((np.sum(df['weight_fixed model']**2))/(np.sum(df['weight_fixed model'])))
            degf= len(df.index)-1
            if qq <= degf or degf==0:
                I2=0
                t2=0
            else:
                t2=(qq-degf)/c
                I2=(qq-degf)/qq

            df["weight_random model"]= 1/((df['V']**2)+t2)
            df['LnRR*w_random']=df['weight_random model']*df['LnRR']
            df['weight(%)_random model']=100*df['weight_random model']/np.sum(df['weight_random model'])
            df['weight(%)_fixed model']=100*df['weight_fixed model']/np.sum(df['weight_fixed model'])

            LnRR_total_random=np.sum(df['LnRR*w_random'])/np.sum(df["weight_random model"])
            se_total_random=(1/np.sum(df["weight_random model"]))**0.5
            lower_LnRR_random=LnRR_total_random-1.96*se_total_random
            upper_LnRR_random=LnRR_total_random+1.96*se_total_random
            LnRR_total_fixed=np.sum(df['LnRR*w_fixed'])/np.sum(df["weight_fixed model"])
            se_total_fixed=(1/np.sum(df["weight_fixed model"]))**0.5
            lower_LnRR_fixed=LnRR_total_fixed-1.96*se_total_fixed
            upper_LnRR_fixed=LnRR_total_fixed+1.96*se_total_fixed
            RRave_random=np.exp(LnRR_total_random)
            lower_RRave_random=np.exp(lower_LnRR_random)
            upper_RRave_random=np.exp(upper_LnRR_random)
            RRave_fixed=np.exp(LnRR_total_fixed)
            lower_RRave_fixed=np.exp(lower_LnRR_fixed)
            upper_RRave_fixed=np.exp(upper_LnRR_fixed)
            z_score_fixed=LnRR_total_fixed/se_total_fixed
            p_value_fixed = scipy.stats.norm.sf(abs(z_score_fixed))*2
            z_score_random=LnRR_total_random/se_total_random
            p_value_random = scipy.stats.norm.sf(abs(z_score_random))*2

            #moderator-regression analysis
            moderator_=df['Moderator']
            effect_size=df['LnRR']
            moderator_ = sm.add_constant(moderator_)
            model = sm.OLS(effect_size, moderator_).fit()
            predictions = model.predict(moderator_)
            results=model.summary()
            moder=results.as_html()


            df.drop(['LnRR*w_fixed','LnRR2*w_fixed','weight_random model','LnRR*w_random','weight_fixed model', 'd'] ,inplace=True, axis=1)
            df.columns = ['Study name', 'Events-g1', 'Non-Events_g1' , 'Events-g2', 'Non-Events_g2' ,'Moderator', 'RiskRatio', 'LnRR' , 'V', 'SE', 'lower_lnRR', 'upper_lnRR', 'lower_RR', 'upper_RR', 'weight(%)_random model', 'weight(%)_fixed model' ]
            df.index+=1

            df2=df.to_dict(orient="dict")
            writer = pd.ExcelWriter('results/MetaMar_result_ratioxl.xlsx')
            df.to_excel(writer,'Sheet1')
            writer.save()

            study_list = list(map(lambda x: str(x), df['Study name'].tolist()))
            resultData = {
                'result_table': HTML(df.to_html(classes="responsive-table-2 rt cf")),
                'study_list': study_list,
                'RR_list': df['RiskRatio'].tolist(),
                'lower_RR_list': df['lower_RR'].tolist(),
                'upper_RR_list': df['upper_RR'].tolist(),
                'RR_weight_list': df['weight(%)_fixed model'].tolist(),
                'LnRR_total_random': float("{0:.2f}".format(LnRR_total_random)),
                'RRave_random': float("{0:.2f}".format(RRave_random)),
                'se_total_random': float("{0:.3f}".format(se_total_random)),
                'lower_RRave_random': float("{0:.3f}".format(lower_RRave_random)),
                'upper_RRave_random': float("{0:.3f}".format(upper_RRave_random)),
                'Het_random': 100*float("{0:.3f}".format(I2)),
                't2': float("{0:.3f}".format(t2)),
                'LnRR_total_fixed': float("{0:.2f}".format(LnRR_total_fixed)),
                'RRave_fixed': float("{0:.2f}".format(RRave_fixed)),
                'se_total_fixed': float("{0:.3f}".format(se_total_fixed)),
                'lower_RRave_fixed': float("{0:.3f}".format(lower_RRave_fixed)),
                'upper_RRave_fixed': float("{0:.3f}".format(upper_RRave_fixed)),
                'Het_fixed': 100*float("{0:.3f}".format(I2)),
                'p_value_fixed': float("{0:.6f}".format(p_value_fixed)),
                'p_value_random': float("{0:.6f}".format(p_value_random)),
                'z_score_fixed': float("{0:.3f}".format(z_score_fixed)),
                'z_score_random': float("{0:.3f}".format(z_score_random)),
                'moder': moder
            }

            return render_template("result_ratiosxls.html", **resultData)



        except Exception as error:
            print(error)
            return render_template('error_content.html')



@app.route('/return-file-ratioxl/')
def return_file_ratioxl():
    return send_file('results/MetaMar_result_ratioxl.xlsx', attachment_filename='MetaMar_result_ratioxl.xlsx')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/meta')
def meta():
    return render_template('select.html')

@app.route('/smd')
def smd():
    return render_template('meta.html')

@app.route('/corr')
def corr():
    return render_template('correlation.html')

@app.route('/ratios')
def odds():
    return render_template('ratios.html')

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
