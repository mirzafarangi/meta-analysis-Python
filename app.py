from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import copy
from IPython.display import display, HTML

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/_method2')
def method2():
    return render_template('method2.html')

@app.route('/_contact')
def contact():
    return render_template('contact.html')

@app.route('/_analysis1')
def analysis1():


    study=request.args.get('study')


    g1_sample=float(request.args.get('g1_sample'))
    g1_mean=float(request.args.get('g1_mean'))
    g1_sd=float(request.args.get('g1_sd'))

    g2_sample=float(request.args.get('g2_sample'))
    g2_mean=float(request.args.get('g2_mean'))
    g2_sd=float(request.args.get('g2_sd'))

    SE=(((g1_sample-1)*g1_sd**2+(g2_sample-1)*g2_sd**2)/(g1_sample+g2_sample-2))**0.5

    d=float("{0:.2f}".format((g2_mean-g1_mean)/SE))
    g=float("{0:.2f}".format(d*(1-(3/(4*(g1_sample+g2_sample)-9)))))

    n=g1_sample+g2_sample
    n_1=(1/g1_sample)+(1/g2_sample)

    SEd=float("{0:.2f}".format(((n_1+(d**2/(2*n))))**0.5))
    SEg=float("{0:.2f}".format(SEd*(1-3/(4*(n)-9))))

    d_lower=float("{0:.2f}".format(d-1.96*SEd))
    d_upper=float("{0:.2f}".format(d+1.96*SEd))

    g_lower=float("{0:.2f}".format(g-1.96*SEg))
    g_upper=float("{0:.2f}".format(g+1.96*SEg))


    return jsonify(study=study, d=d, SEd=SEd, d_lower=d_lower, d_upper=d_upper, g=g, SEg=SEg, g_lower=g_lower, g_upper=g_upper)

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





      return render_template("analysis.html", total=HTML(df.to_html()), ave_d=float("{0:.2f}".format(d_total)), ave_SEd=float("{0:.2f}".format(s_total)),lower_dd=float("{0:.2f}".format(lower_d)),upper_dd=float("{0:.2f}".format(upper_d)),ave_g=float("{0:.2f}".format(g_total)), ave_SEg=float("{0:.3f}".format(sg_total)),lower_gg=float("{0:.3f}".format(lower_g)),upper_gg=float("{0:.3f}".format(upper_g)), Het=100*float("{0:.3f}".format(I2)))

if __name__ == '__main__':
    app.run()
