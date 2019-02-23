# New limb enhancer discovery algorithm, part 1.
# 
# - After this program runs, calcualte RPKM inputs of segments: run bedToRPKM.py on the output of this program.
#   bedToRPKM.py outputs the features for these segments
# - Then score the segments with predict16dComboModel.py. Analyze the scores and merge segments.
# 
# Find candidate segments in chromosome chr9 with positive DNase Z-scores but not in known enhancer regions.
# Z-Scores max is limited to 7.5 since the max for all enhancers is 7.11.
#
# To do this, the RPKM values are found for DNase for each segment. Segments are 1kb long and overlap by 0.5kb
# Output is written tot a TSV file.


bedGraphFileName = '../DataProcessing/bedGraphDataFiles/Chromatin/DNase1-HindLimbRep2-ENCBS752MTX_wgEncodeUwDnaseHlbudCd1ME11halfSigRep2mm10.bedGraph'
chromosomeReadValue = 'chr9'
outFileName ='Chr9CandidateSegments.tsv'
# Mean for the data (Z-Score = 0) is 7996
minVal = 8000
# Limit Z-Score to 7.5. because max Z-Score for all enhancers in data is 7.11
# Std deviation is 8650
maxVal = minVal + 7.5 * 8650 

# Known enhancers in chr9, sorted by start/end base pair coordinates
enhStart = [13697970, 21556730, 23377927, 24720490, 24737086, 24748244, 24754514, 24789002, 24802515, 24811553, 24833559, 
24860616, 24887072, 24919749, 24922167, 24973917, 25055368, 27937330, 29318640, 34557590, 35423877, 37053339, 41071639, 
41735464, 43564181, 44877225, 46118249, 46381148, 46620224, 46660108, 47376512, 49736440, 52147925, 54335032, 57806611, 
61370059, 61600844, 62414649, 63108235, 63131587, 63176077, 63523991, 63880199, 64044784, 65381642, 65825989, 67001298, 
67040666, 68794558, 70834536, 71974987, 71975968, 88030648, 88048577, 88070457, 88136652, 88195357, 90125755, 90313788, 
90419622, 90691310, 90692810, 90700206, 90730671, 90739783, 90833187, 90902261, 91073326, 91087949, 91366859, 91445768, 
91462358, 95812717, 96380312, 99773255, 99994346, 100130872, 100166423, 100194021, 100210152, 100271529, 103679109, 
106382995, 106416951, 107305936, 109892964, 118317971, 118858597, 118863536, 119496374, 120601909, 121301588, 121355289]

enhEnd = [13700760, 21559920, 23379852, 24721595, 24737801, 24749108, 24755359, 24789893, 24803444, 24812154, 24834351, 
24861316, 24890068, 24921117, 24923116, 24975039, 25056389, 27938180, 29319071, 34559391, 35427018, 37055278, 41074687, 
41736837, 43568174, 44880193, 46121546, 46381839, 46623042, 46660820, 47377360, 49738293, 52150393, 54337557, 57807834, 
61371070, 61603666, 62415677, 63110238, 63133435, 63177276, 63524533, 63881508, 64046949, 65384657, 65827111, 67004558, 
67043521, 68797064, 70835234, 71975901, 71976731, 88034978, 88050948, 88074527, 88139609, 88199255, 90126865, 90316367, 
90421316, 90692526, 90693923, 90701273, 90732594, 90741475, 90835006, 90905638, 91074669, 91090092, 91368203, 91447285, 
91463892, 95815609, 96382941, 99774439, 99995440, 100131771, 100168226, 100195330, 100211938, 100272974, 103679657, 
106385663, 106418793, 107309959, 109896468, 118319794, 118861666, 118868147, 119501352, 120604533, 121305883, 121357466]

segmentLeft = (enhStart[0] // 1000) * 1000 # 13600000
segmentRight = segmentLeft + 1000
end = 121500000
lastWritten = 0
segmentScore = 0.0
nextOverlappingSegmentScore = 0.0
nextOverlappingSegmentLeft = segmentLeft + 500
nextOverlappingSegmentRight = nextOverlappingSegmentLeft + 1000
linelist = open(bedGraphFileName)
outfile = open(outFileName, 'w')
segmentIndex = 0
outfile.write('#Chromosome\tLeft\tRight\tSegment\tClass\tIndex\tDNaseScore\n')
for line in linelist:
  if line.startswith('#') or line.startswith('track'):
    continue
  fields=line.strip().split('\t')
  chr = fields[0]
  if chr not in chromosomeReadValue:
    continue
  left = int(fields[1])
  right = int(fields[2])
  score = float(fields[3])

  if left >= end: break

  # Skip over known enhancer regions. 
  i = len(enhStart) - 1
  while i >= 0 and left < enhStart[i]: i = i - 1
  if i >=  0:
    if left < enhEnd[i]: left = enhEnd[i]
    if right < enhEnd[i]: continue
    # Note: If right > enhEnd[i+1] we don't process the part to the right of enhEnd[i+1]
    if i < len(enhStart) - 1 and right > enhStart[i+1]: right = enhStart[i+1]

  segmentScore = segmentScore + max(0.0, min(right, segmentRight) - max(left, segmentLeft)) * score
  nextOverlappingSegmentScore = nextOverlappingSegmentScore + score * max(0.0,
        min(right, nextOverlappingSegmentRight) - max(left, nextOverlappingSegmentLeft))
  if right >= segmentRight:
    # Finished segment
    if segmentScore >= minVal and segmentScore <= maxVal:
      if segmentLeft > lastWritten and lastWritten > 0:
        # Extend previously written segment block by 1 overlapping segment
        segmentIndex = segmentIndex + 1
        outfile.write('chr9\t' + str(lastWritten-500) + '\t' + str(lastWritten+500) + '\t' + 
            'Seg' + str(segmentIndex) + '\tPredict\t' + str(segmentIndex) + '\n')
        lastWritten = lastWritten + 500
      if segmentLeft > lastWritten:
        # Extend segment block to be written by 1 overlapping segment
        segmentIndex = segmentIndex + 1
        lastWritten = segmentLeft
        outfile.write('chr9\t' + str(lastWritten-500) + '\t' + str(lastWritten+500) + '\t' + 
            'Seg' + str(segmentIndex) + '\tPredict\t' + str(segmentIndex) + '\n')
        lastWritten = lastWritten + 500
      segmentIndex = segmentIndex + 1
      # To convert to RPM, we have to multiply segmentScore by the segment length and divide 
      # by 1000. Since segment length is 1000, the RPM is segmentScore
      outfile.write('chr9\t' + str(segmentLeft) + '\t' + str(segmentRight) + '\t' + 
          'Seg' + str(segmentIndex) + '\tPredict\t' + str(segmentIndex) + '\t' + 
           str(segmentScore) + '\n')
      lastWritten = segmentRight
    # Start a new segment
    if right < nextOverlappingSegmentRight:
      segmentLeft = nextOverlappingSegmentLeft
      nextOverlappingSegmentLeft = segmentRight
      segmentRight = nextOverlappingSegmentRight
      nextOverlappingSegmentRight = nextOverlappingSegmentLeft + 1000
      segmentScore = nextOverlappingSegmentScore
      nextOverlappingSegmentScore = score * max(0.0,
          min(right, nextOverlappingSegmentRight) - max(left, nextOverlappingSegmentLeft))
    else:
      # New segment is right of nextOverlappingSegment, so write out nextOverlappingSegment and 
      # start two new segments using left
      if nextOverlappingSegmentScore >= minVal and nextOverlappingSegmentScore <= maxVal:
        # To convert to RPM, we have to multiply segmentScore by the segment length and divide 
        # by 1000. Since segment length is 1000, the RPM is segmentScore
        segmentIndex = segmentIndex + 1
        outfile.write('chr9\t' + str(nextOverlappingSegmentLeft) + '\t' + 
                      str(nextOverlappingSegmentRight) + '\t' + 'Seg' + str(segmentIndex) + 
                      '\tPredict\t' + str(segmentIndex) + '\t' + 
                      str(nextOverlappingSegmentScore) + '\n')
        lastWritten = nextOverlappingSegmentRight
      segmentLeft = (left // 1000) * 1000
      segmentRight = segmentLeft + 1000
      while segmentRight <= right:
        segmentScore = (segmentRight - max(left, segmentLeft)) * score
        # Write the segment
        if segmentScore >= minVal and segmentScore <= maxVal:
          if segmentLeft > lastWritten and lastWritten > 0:
            # Extend previously written segment block by 1 overlapping segment
            segmentIndex = segmentIndex + 1
            outfile.write('chr9\t' + str(lastWritten-500) + '\t' + str(lastWritten+500) + '\t' +
                          'Seg' + str(segmentIndex) + '\tPredict\t' + str(segmentIndex) + '\n')
            lastWritten = lastWritten + 500
          if segmentLeft > lastWritten:
            # Extend segment block to be written by 1 overlapping segment
            segmentIndex = segmentIndex + 1
            lastWritten = segmentLeft
            outfile.write('chr9\t' + str(lastWritten-500) + '\t' + str(lastWritten+500) + '\t' +
                          'Seg' + str(segmentIndex) + '\tPredict\t' + str(segmentIndex) + '\n')
            lastWritten = lastWritten + 500
          segmentIndex = segmentIndex + 1
          # To convert to RPM, we have to multiply segmentScore by the segment length and divide by 1000. 
          # Since segment length is 1000, the RPM is segmentScore
          outfile.write('chr9\t' + str(segmentLeft) + '\t' + str(segmentRight) + '\t' + 'Seg' +
                        str(segmentIndex) + '\tPredict\t' + str(segmentIndex) + '\t' + 
                        str(segmentScore) + '\n')
          lastWritten = segmentLeft
        segmentLeft = segmentLeft + 500
        segmentRight = segmentRight + 500
      # End of while
      segmentScore = max(0.0,  min(right, segmentRight) - max(left, segmentLeft)) * score
      nextOverlappingSegmentLeft = segmentLeft + 500
      nextOverlappingSegmentRight = nextOverlappingSegmentLeft + 1000
      nextOverlappingSegmentScore = score * max(0.0,
          min(right, nextOverlappingSegmentRight) - max(left, nextOverlappingSegmentLeft))
 
outfile.close()
