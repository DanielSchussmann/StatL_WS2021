import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
# data_draw=pd.read_csv('/content/drive/MyDrive/seminar_data/AUD_NZD.csv',usecols=[1,2,3,4])[:260]
data_read = pd.read_csv('market_data/EUR_USD_D.csv', usecols=[0,1, 2, 3, 4])
#data_read = pd.read_csv('market_data/EUR_USD_D.csv', usecols=[1, 2, 3, 4])
app = dash.Dash(__name__)

fig_XABCD = go.Figure(data=[go.Candlestick(
    open=data_read['Open'],
    high=data_read['High'],
    low=data_read['Low'],
    close=data_read['Close'], name='price', opacity=0.8)], layout_title_text='Trades overview')

SMA= lambda n,data:[np.sum(data[x-n:x])/n for x in range(len(data),n,-1)]

sma_10 = SMA(10,data_read['Close'])
sma_50 = SMA(50,data_read['Close'])
#print(sma_10)
fig_XABCD.add_trace(go.Scatter(x=np.arange(len(data_read['Close']),10,-1), y=sma_10, name='SMA_10', line=dict(color='blue')))  # lines
fig_XABCD.add_trace(go.Scatter(x=np.arange(len(data_read['Close']),50,-1), y=sma_50, name='SMA_50', line=dict(color='orange')))  # lines



def XABCD(data, ret_ext):
    open = np.array(data['Open'])
    close = np.array(data['Close'])
    high = np.array(data['High'])
    low = np.array(data['Low'])

    XB = ret_ext[0]
    AC = ret_ext[1]
    BD = ret_ext[2]
    XD = ret_ext[3]
    dist = 10
    output = []

    for x in range(1, len(data) - dist * 2):
        X_o = open[x]
        X_h = high[x]
        X_l = low[x]
        X_c = close[x]
        X = [X_o, X_h, X_l, X_c]

        for a in range(x + 1, x + dist):
            A_o = open[a]
            A_h = high[a]
            A_l = low[a]
            A_c = close[a]
            A = [A_o, A_h, A_l, A_c]

            B_top = -(XB[0] - 1) * (A_h - X_h) + X_h
            B_bot = -(XB[1] - 1) * (A_l - X_l) + X_l

            C_top = AC[1] * (A_h - B_top) + B_top  # (C-B)/(A-B)
            C_bot = AC[0] * (A_l - B_bot) + B_bot

            D_top = -(XD[0] - 1) * (A_h - X_h) + X_h
            D_bot = -(XD[1] - 1) * (A_l - X_l) + X_l
            # print(B_top,B_bot,C_top,C_bot,D_top,D_bot)
            b = np.where(np.logical_and(B_bot <= close[a + 1:a + dist], close[a + 1:a + dist] <= B_top))[0]
            c = np.where(np.logical_and(C_bot <= close[a + 1:a + dist], close[a + 1:a + dist] <= C_top))[0]
            d = np.where(np.logical_and(D_bot <= close[a + 1:a + dist], close[a + 1:a + dist] <= D_top))[0]

            if b.size != 0 and c.size != 0 and d.size != 0:
                c_bigger = np.array(np.where(c > b[-1]))
                b = b + a + 1

                if c_bigger.size != 0:
                    c = c[c_bigger[0]]
                    d_bigger = np.array(np.where(d > c[-1]))

                    if d_bigger.size != 0:
                        c = c + a + 1
                        d = d[d_bigger[0]] + a + 1

                        for i_b in range(len(b)):
                            for i_c in range(len(c)):
                                for i_d in range(len(d)):
                                    B = close[b[i_b]]
                                    C = close[c[i_c]]
                                    D = close[d[i_d]]
                                    i_BD = (D - C) / (B - C)

                                    if BD[0] <= i_BD <= BD[1]:
                                        for i_x in range(len(X)):
                                            for i_a in range(len(A)):
                                                i_XB =1- (B - X[i_x]) / (A[i_a] - X[i_x])

                                                if XB[0] <= i_XB <= XB[1]:
                                                    i_AC = (C - B) / (A[i_a] - B)

                                                    if AC[0] <= i_AC <= AC[1]:
                                                        i_XD = 1 - (D - X[i_x]) / (A[i_a] - X[i_x])

                                                        if XD[0] <= i_XD <= XD[1]:
                                                            # print(i_XD)
                                                            output.append([[x, a, b[i_b], c[i_c], d[i_d]],
                                                                           [X[i_x], A[i_a], B, C, D],
                                                                           [i_XB, i_AC, i_BD, i_XD]])





    return output


# print(XABCD(data_read))
def add_XABCD(data, col, bez):
    x, a, b, c, d, X, A, B, C, D, XB, AC, BD, XD = data[0][0], data[0][1], data[0][2], data[0][3], data[0][4], data[1][0], data[1][1], data[1][2], data[1][3], data[1][4], data[2][0], data[2][1], data[2][2], data[2][3]

    fig_XABCD.add_trace(go.Scatter(x=[x, a, b, None, b, c, d], y=[X, A, B, None, B, C, D], name=bez, line=dict(color=col)))  # lines
    fig_XABCD.add_trace(go.Scatter(x=[x, b, None, x, d, None, a, c, None, b, d], name=bez, y=[X, B, None, X, D, None, A, C, None, B, D],line=dict(width=1, dash='dash', color=col)))  # dash connections
    fig_XABCD.add_annotation(name="X", x=x, y=X, yshift=-13, font_color="white", bgcolor=col, text='X', showarrow=False,font_size=10)
    fig_XABCD.add_annotation(name="A", x=a, y=A, yshift=13, font_color="white", bgcolor=col, text='A', showarrow=False,font_size=10)
    fig_XABCD.add_annotation(name="B", x=b, y=B, yshift=-13, font_color="white", bgcolor=col, text='B', showarrow=False, font_size=10)
    fig_XABCD.add_annotation(name="C", x=c, y=C, yshift=13, font_color="white", bgcolor=col, text='C', showarrow=False, font_size=10)
    fig_XABCD.add_annotation(name="D", x=d, y=D, yshift=-13, font_color="white", bgcolor=col, text='D', showarrow=False,font_size=10)
    fig_XABCD.add_annotation(name="type", x=b, y=C, yshift=43, font_color=col, bgcolor='white', text=bez,showarrow=False, font_size=14)
    fig_XABCD.add_annotation(name="XB", x=(x + b) / 2, y=(X + B) / 2, font_color="white", bgcolor=col,text=round(XB, 3), showarrow=False, font_size=10)
    fig_XABCD.add_annotation(name="AC", x=(a + c) / 2, y=(A + C) / 2, font_color="white", bgcolor=col, text=round(AC, 3), showarrow=False, font_size=10)
    fig_XABCD.add_annotation(name="BD", x=(b + d) / 2, y=(B + D) / 2, font_color="white", bgcolor=col,text=round(BD, 3), showarrow=False, font_size=10)
    fig_XABCD.add_annotation(name="XD", x=(x + d) / 2, y=(X + D) / 2, font_color="white", bgcolor=col,text=round(XD, 3), showarrow=False, font_size=10)


Crab = XABCD(data_read, [[0.382, 0.618], [0.382, 0.886], [2.24, 3.618], [1.56000, 1.69]])
# gartley [[0.61, 0.618],[0.382, 0.886],[1.13, 1.618],[0.78,0.786]]
Gartley = XABCD(data_read, [[0.56, 0.618], [0.382, 0.886], [1.13, 1.618], [0.78, 0.786]])
Bat = XABCD(data_read, [[0.382, 0.500], [0.382, 0.886], [1.618, 2.618], [0.88, 0.89]])
Butterfly = XABCD(data_read, [[0.77,0.79], [0.382, 0.886], [1.618, 2.618], [1.27, 1.618]])
Shark = XABCD(data_read, [[0.382, 0.500], [0.382, 0.886], [1.618, 2.618], [0.88, 0.89]])
# print(results)
for r in range(len(Bat)):
        add_XABCD(Bat[r], 'purple', 'crab')
for r in range(len(Crab)):
        add_XABCD(Crab[r], 'red', 'Gartley')
for r in range(len(Gartley)):
        add_XABCD(Gartley[r], 'pink', 'Bat')
for r in range(len(Butterfly)):
        add_XABCD(Butterfly[r], 'coral', 'Butterlfy')



fig_XABCD.update_layout(xaxis_rangeslider_visible=False, showlegend=False)
# fig_XABCD.update_annotations(hoverlabel=dict(font_color="white",bgcolor='Royalblue'),hovertext='name')
app.layout = dash.html.Div(children=[
    dash.html.H1(children='HARMONIC SCANNER', style={'textAlign': 'center', 'margin-bottom': '10px'}),
    dash.dcc.Graph(id='F_O_V', figure=fig_XABCD, style={'fontsize': '30px', 'width': '98vw', 'height': '85vh', 'margin-bottom': '5px'}),])

if __name__ == '__main__':
    app.run_server(debug=True)