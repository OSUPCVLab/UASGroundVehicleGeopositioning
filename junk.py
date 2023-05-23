ls = ['CF_KP_is_Lo_dists.png',
'CF_RF_KP_is_SG_and_Lo_dists.png',
'NF_KP_is_SG_dists.png',
'CF_KP_is_Lo_meanDists.png',
'CF_RF_KP_is_SG_and_Lo_meanDists.png',
'NF_KP_is_SG_meanDists.png',
'CF_KP_is_SG_and_Lo_dists.png',
'CF_RF_KP_is_SG_dists.png',
'RF_KP_is_Lo_dists.png',
'CF_KP_is_SG_and_Lo_meanDists.png',
'CF_RF_KP_is_SG_meanDists.png',
'RF_KP_is_Lo_meanDists.png',
'CF_KP_is_SG_dists.png',
'NF_KP_is_Lo_dists.png',
'RF_KP_is_SG_and_Lo_dists.png',
'CF_KP_is_SG_meanDists.png',
'NF_KP_is_Lo_meanDists.png',
'RF_KP_is_SG_and_Lo_meanDists.png',
'CF_RF_KP_is_Lo_dists.png',
'NF_KP_is_SG_and_Lo_dists.png',
'RF_KP_is_SG_dists.png',
'CF_RF_KP_is_Lo_meanDists.png',
'NF_KP_is_SG_and_Lo_meanDists.png',
'RF_KP_is_SG_meanDists.png']
ls.sort()
h=3.75
print()
for i in range(int(len(ls)/4)):
    
    print(f'\centerline{{ \n\includegraphics[height={h}cm]{{images/{ls[i*2]}}} \n\includegraphics[height={h}cm]{{images/{ls[i*2+1]}}} \hspace{{5mm}} \n\includegraphics[height={h}cm]{{images/{ls[i*2+2]}}} \n\includegraphics[height={h}cm]{{images/{ls[i*2+3]}}} }}')
    
print()
