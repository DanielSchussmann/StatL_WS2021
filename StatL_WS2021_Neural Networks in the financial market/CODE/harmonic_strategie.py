import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from backtest import *




def XABCD_fixed_bull(data, ret_ext):
    open = np.array(data['Open'])
    close = np.array(data['Close'])
    high = np.array(data['High'])
    low = np.array(data['Low'])

    candle = np.array([open, high, low, close], dtype=object)

    XB = ret_ext[0]
    AC = ret_ext[1]
    BD = ret_ext[2]
    XD = ret_ext[3]
    dist = 8
    output = []
    all_patterns = []
    D_prices = np.array([open[-1], high[-1], low[-1], close[-1]])
    D = len(data) - 1
    for x in range(1, len(data) - 4):
        i_X = [open[x], high[x], low[x], close[x]]

        for i_x in range(len(i_X)):
            X = i_X[i_x]

            for a in range(x + 1, len(data) - 3):
                i_A = [open[a], high[a], low[a], close[a]]

                for i_a in range(len(i_A)):
                    A = i_A[i_a]

                    if X - A != 0:
                        B_top = (-(XB[0] - 1) * (A - X) + X) if X < A else (XB[1] * (X - A) + A)
                        B_bot = (-(XB[1] - 1) * (A - X) + X) if X < A else (XB[0] * (X - A) + A)

                        C_top = (AC[1] * (A - B_top) + B_top) if X < A else (AC[1] * (B_top - A) + A)
                        C_bot = (AC[0] * (A - B_bot) + B_bot) if X < A else (AC[0] * (B_bot - A) + A)

                        b_c = np.where(np.logical_and(B_bot <= close[a + 1:-2], close[a + 1:-2] <= B_top))
                        b_h = np.where(np.logical_and(B_bot <= high[a + 1:-2], high[a + 1:-2] <= B_top))
                        b_l = np.where(np.logical_and(B_bot <= low[a + 1:-2], low[a + 1:-2] <= B_top))
                        b_o = np.where(np.logical_and(B_bot <= open[a + 1:-2], open[a + 1:-2] <= B_top))
                        b = np.asarray((b_o[0] + a + 1, b_h[0] + a + 1, b_l[0] + a + 1, b_c[0] + a + 1), dtype=object)
                        b_price = np.asarray((candle[0][b[0].astype(int)], candle[1][b[1].astype(int)],
                                              candle[2][b[2].astype(int)], candle[3][b[3].astype(int)]), dtype=object)

                        if b.size != 0:
                            c_c = np.where(np.logical_and(C_bot <= close[a + 2:-2], close[a + 2:-2] <= C_top))
                            c_h = np.where(np.logical_and(C_bot <= high[a + 2:-2], high[a + 2:-2] <= C_top))
                            c_l = np.where(np.logical_and(C_bot <= low[a + 2:-2], low[a + 2:-2] <= C_top))
                            c_o = np.where(np.logical_and(C_bot <= open[a + 2:-2], open[a + 2:-2] <= C_top))
                            c = np.asarray((c_o[0] + a + 2, c_h[0] + a + 2, c_l[0] + a + 2, c_c[0] + a + 2),
                                           dtype=object)

                            if c.size != 0:
                                for b_o in range(0, 4):
                                    for b_i in range(len(b[b_o])):
                                        # print(x,i_x,a,i_a,b_o,b_i,b[b_o][b_i],b_price[b_o][b_i])
                                        B = b[b_o][b_i]
                                        B_price = b_price[b_o][b_i]
                                        i_C = np.array([c[i][np.where(c[i] > B)] for i in range(len(c))], dtype=object)

                                        if i_C.size != 0:
                                            i_C_price = np.asarray((candle[0][i_C[0].astype(int)],
                                                                    candle[1][i_C[1].astype(int)],
                                                                    candle[2][i_C[2].astype(int)],
                                                                    candle[3][i_C[3].astype(int)]), dtype=object)
                                            i_AC = (i_C_price - B_price) / (A - B_price)
                                            i_C = np.array(
                                                [i_C[i][np.where(np.logical_and(AC[0] <= i_AC[i], i_AC[i] <= AC[1]))]
                                                 for i in range(len(i_C))], dtype=object)

                                            if i_C.size != 0:
                                                for c_o in range(0, 4):
                                                    for c_i in range(len(i_C[c_o])):
                                                        C = i_C[c_o][c_i]
                                                        C_price = candle[c_o][C]
                                                        i_BD = (D_prices - C_price) / (B_price - C_price)
                                                        i_XD = 1 - (D_prices - X) / (A - X)

                                                        if BD[0] <= i_BD[3] <= BD[1] and XD[0] <= i_XD[3] <= XD[1]:
                                                            D_price = D_prices[3]
                                                            i_XB = 1 - (B_price - X) / (A - X)
                                                            i_BD = (D_price - C_price) / (B_price - C_price)
                                                            i_AC = (C_price - B_price) / (A - B_price)
                                                            i_XD = 1 - (D_price - X) / (A - X)

                                                            if len(output) == 0:
                                                                output.append(
                                                                    [[x, a, B, C, D], [X, A, B_price, C_price, D_price],
                                                                     [i_XB, i_AC, i_BD, i_XD]])
                                                            elif x != output[-1][0][0]:
                                                                output.append(
                                                                    [[x, a, B, C, D], [X, A, B_price, C_price, D_price],
                                                                     [i_XB, i_AC, i_BD, i_XD]])
                                                                all_patterns.append(
                                                                    [[x, a, B, C, D], [X, A, B_price, C_price, D_price],
                                                                     [i_XB, i_AC, i_BD, i_XD]])
                                                            else:
                                                                all_patterns.append(
                                                                    [[x, a, B, C, D], [X, A, B_price, C_price, D_price],
                                                                     [i_XB, i_AC, i_BD, i_XD]])
                                                            break


                                                        elif BD[0] <= i_BD[2] <= BD[1] and XD[0] <= i_XD[2] <= XD[1]:
                                                            D_price = D_prices[2]
                                                            i_XB = 1 - (B_price - X) / (A - X)
                                                            i_BD = (D_price - C_price) / (B_price - C_price)
                                                            i_AC = (C_price - B_price) / (A - B_price)
                                                            i_XD = 1 - (D_price - X) / (A - X)

                                                            if len(output) == 0:
                                                                output.append(
                                                                    [[x, a, B, C, D], [X, A, B_price, C_price, D_price],
                                                                     [i_XB, i_AC, i_BD, i_XD]])
                                                            elif x != output[-1][0][0]:
                                                                output.append(
                                                                    [[x, a, B, C, D], [X, A, B_price, C_price, D_price],
                                                                     [i_XB, i_AC, i_BD, i_XD]])
                                                                all_patterns.append(
                                                                    [[x, a, B, C, D], [X, A, B_price, C_price, D_price],
                                                                     [i_XB, i_AC, i_BD, i_XD]])
                                                            else:
                                                                all_patterns.append(
                                                                    [[x, a, B, C, D], [X, A, B_price, C_price, D_price],
                                                                     [i_XB, i_AC, i_BD, i_XD]])
                                                            break


                                                        elif BD[0] <= i_BD[1] <= BD[1] and XD[0] <= i_XD[1] <= XD[1]:
                                                            D_price = D_prices[1]
                                                            i_XB = 1 - (B_price - X) / (A - X)
                                                            i_BD = (D_price - C_price) / (B_price - C_price)
                                                            i_AC = (C_price - B_price) / (A - B_price)
                                                            i_XD = 1 - (D_price - X) / (A - X)

                                                            if len(output) == 0:
                                                                output.append(
                                                                    [[x, a, B, C, D], [X, A, B_price, C_price, D_price],
                                                                     [i_XB, i_AC, i_BD, i_XD]])
                                                            elif x != output[-1][0][0]:
                                                                output.append(
                                                                    [[x, a, B, C, D], [X, A, B_price, C_price, D_price],
                                                                     [i_XB, i_AC, i_BD, i_XD]])
                                                                all_patterns.append(
                                                                    [[x, a, B, C, D], [X, A, B_price, C_price, D_price],
                                                                     [i_XB, i_AC, i_BD, i_XD]])
                                                            else:
                                                                all_patterns.append(
                                                                    [[x, a, B, C, D], [X, A, B_price, C_price, D_price],
                                                                     [i_XB, i_AC, i_BD, i_XD]])
                                                            break


                                                        elif BD[0] <= i_BD[0] <= BD[1] and XD[0] <= i_XD[0] <= XD[1]:
                                                            D_price = D_prices[0]
                                                            i_XB = 1 - (B_price - X) / (A - X)
                                                            i_BD = (D_price - C_price) / (B_price - C_price)
                                                            i_AC = (C_price - B_price) / (A - B_price)
                                                            i_XD = 1 - (D_price - X) / (A - X)

                                                            if len(output) == 0:
                                                                output.append(
                                                                    [[x, a, B, C, D], [X, A, B_price, C_price, D_price],
                                                                     [i_XB, i_AC, i_BD, i_XD]])
                                                            elif x != output[-1][0][0]:
                                                                output.append(
                                                                    [[x, a, B, C, D], [X, A, B_price, C_price, D_price],
                                                                     [i_XB, i_AC, i_BD, i_XD]])
                                                                all_patterns.append(
                                                                    [[x, a, B, C, D], [X, A, B_price, C_price, D_price],
                                                                     [i_XB, i_AC, i_BD, i_XD]])
                                                            else:
                                                                all_patterns.append(
                                                                    [[x, a, B, C, D], [X, A, B_price, C_price, D_price],
                                                                     [i_XB, i_AC, i_BD, i_XD]])
                                                            break


                                                else:
                                                    continue
                                                break
                                    else:
                                        continue
                                    break
                                else:
                                    continue
                                break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue

        break

    return output
"""
harm=BACKTEST_v3()


[harm.tick() for x in range(0,30)]

my_orders=[]
for x in range(0,105):
    if x%15==0:
        for s in range(len(harm.symbols)):
            data_read=harm.data[harm.symbols[s]][harm.index-15:harm.index]
            Crab = XABCD_fixed_bull(data_read, [[0.382, 0.618], [0.382, 0.886], [2.24, 3.618], [1.6175, 1.6185]])
            Gartley = XABCD_fixed_bull(data_read, [[0.6175, 0.6185], [0.382, 0.886], [1.13, 1.618], [0.7855, 0.7865]])
            Bat = XABCD_fixed_bull(data_read, [[0.382, 0.500], [0.382, 0.886], [1.618, 2.618], [0.8855, 0.8865]])
            Butterfly = XABCD_fixed_bull(data_read, [[0.7855, 0.7865], [0.382, 0.886], [1.618, 2.24], [1.265, 1.272]])

            if len(Crab) !=0:
                print('Crab')
                xxx = harm.MakeOrder(harm.symbols[s], 'SHORT')
                my_orders.append(xxx)
            elif len(Gartley) !=0:
                print('Gart')
                xxx = harm.MakeOrder(harm.symbols[s], 'SHORT')
                my_orders.append(xxx)
            elif len(Bat) !=0:
                print('Bat')
                xxx = harm.MakeOrder(harm.symbols[s], 'SHORT')
                my_orders.append(xxx)

    o = 0
    while o < len(my_orders):
        if harm.open_orders[my_orders[o]][4] + 5 < harm.index:
            harm.SellOrder(my_orders[o])
            my_orders.pop(o)
        o += 1

    harm.tick()

harm.init_layout()
harm.draw()
"""
# Shark     =  XABCD_fixed_bull(data_read, [[0.382, 0.500], [0.382, 0.886], [1.618, 2.618], [0.8855, 0.8865]])
# print(Shark)
# print(Crab)
"""
for r in range(0, max(len(Bat), len(Crab), len(Gartley), len(Butterfly), len(Shark))):
    if r < len(Bat):
        add_XABC_D(Bat[r], 'purple', 'Bat')
    if r < len(Crab):
        add_XABC_D(Crab[r], 'red', 'Crab')
    if r < len(Gartley):
        add_XABC_D(Gartley[r], 'blue', 'Gartley')
    if r < len(Butterfly):
        add_XABC_D(Butterfly[r], 'coral', 'Butterlfy')
    # if r<len(Shark):
    # add_XABC_D(Shark[r], 'cyan', 'Shark')

fig_XABC_D.update_layout(xaxis_rangeslider_visible=False, showlegend=False)
fig_XABC_D.show()         """

