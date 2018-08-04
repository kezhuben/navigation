from urllib import request, error
import json
import time

inv = 1.0 / 1e6


def decode(encoded):
    decoded = []
    previous = [0, 0]
    i = 0
    # for each byte
    while i < len(encoded):
        # for each coord (lat, lon)
        ll = [0, 0]
        for j in [0, 1]:
            shift = 0
            byte = 0x20
            # keep decoding bytes until you have this coord
            while byte >= 0x20:
                byte = ord(encoded[i]) - 63
                i += 1
                ll[j] |= (byte & 0x1f) << shift
                shift += 5
            # get the final value adding the previous offset and remember it for the next
            ll[j] = previous[j] + (~(ll[j] >> 1) if ll[j] & 1 else (ll[j] >> 1))
            previous[j] = ll[j]
        # scale by the precision and chop off long coords also flip the positions so
        # its the far more standard lon,lat instead of lat,lon
        decoded.append([float('%.6f' % (ll[1] * inv)), float('%.6f' % (ll[0] * inv))])
    # hand back the list of coordinates
    return decoded


try:
    crawl_url = "http://121.46.20.210:8007/route?api_key=valhalla-7UikjOk&json={%22locations%22:[{%22lat%22:23.136532062402154,%22lon%22:113.32118511199953},{%22lat%22:23.11589095262163,%22lon%22:113.28457832336427}],%22costing%22:%22auto%22,%22directions_options%22:{%22units%22:%22km%22,%22language%22:%22zh-CN%22,%22user_intersection_shap%22:%22true%22}}"
    # req = urllib.request(crawl_url)
    response = request.urlopen(crawl_url)
    html = response.read()
except error as e:
    print(e.code)
    print(e.read())

hjson = json.loads(html)
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + " " + str(hjson['trip']['status']) + "\r\n")
edgeidAr = []

# enum TripDirections_Maneuver_Type {
#   TripDirections_Maneuver_Type_kNone = 0,
#   TripDirections_Maneuver_Type_kStart = 1,
#   TripDirections_Maneuver_Type_kStartRight = 2,
#   TripDirections_Maneuver_Type_kStartLeft = 3,
#   TripDirections_Maneuver_Type_kDestination = 4,
#   TripDirections_Maneuver_Type_kDestinationRight = 5,
#   TripDirections_Maneuver_Type_kDestinationLeft = 6,
#   TripDirections_Maneuver_Type_kBecomes = 7,
#   TripDirections_Maneuver_Type_kContinue = 8,
#   TripDirections_Maneuver_Type_kSlightRight = 9,
#   TripDirections_Maneuver_Type_kRight = 10,
#   TripDirections_Maneuver_Type_kSharpRight = 11,
#   TripDirections_Maneuver_Type_kUturnRight = 12,
#   TripDirections_Maneuver_Type_kUturnLeft = 13,
#   TripDirections_Maneuver_Type_kSharpLeft = 14,
#   TripDirections_Maneuver_Type_kLeft = 15,
#   TripDirections_Maneuver_Type_kSlightLeft = 16,
#   TripDirections_Maneuver_Type_kRampStraight = 17,
#   TripDirections_Maneuver_Type_kRampRight = 18,
#   TripDirections_Maneuver_Type_kRampLeft = 19,
#   TripDirections_Maneuver_Type_kExitRight = 20,
#   TripDirections_Maneuver_Type_kExitLeft = 21,
#   TripDirections_Maneuver_Type_kStayStraight = 22,
#   TripDirections_Maneuver_Type_kStayRight = 23,
#   TripDirections_Maneuver_Type_kStayLeft = 24,
#   TripDirections_Maneuver_Type_kMerge = 25,
#   TripDirections_Maneuver_Type_kRoundaboutEnter = 26,
#   TripDirections_Maneuver_Type_kRoundaboutExit = 27,
#   TripDirections_Maneuver_Type_kFerryEnter = 28,
#   TripDirections_Maneuver_Type_kFerryExit = 29,
#   TripDirections_Maneuver_Type_kTransit = 30,
#   TripDirections_Maneuver_Type_kTransitTransfer = 31,
#   TripDirections_Maneuver_Type_kTransitRemainOn = 32,
#   TripDirections_Maneuver_Type_kTransitConnectionStart = 33,
#   TripDirections_Maneuver_Type_kTransitConnectionTransfer = 34,
#   TripDirections_Maneuver_Type_kTransitConnectionDestination = 35,
#   TripDirections_Maneuver_Type_kPostTransitConnectionDestination = 36,
#   TripDirections_Maneuver_Type_kExitStraightRight = 37,
#   TripDirections_Maneuver_Type_kExitStraightLeft = 38,
#   TripDirections_Maneuver_Type_kMulitiWayRightRamp = 39,
#   TripDirections_Maneuver_Type_kMulitiWayLeftRamp = 40,
#   TripDirections_Maneuver_Type_kMulitiWayMiddleRamp = 41,
#   TripDirections_Maneuver_Type_kMulitiWayRight = 42,
#   TripDirections_Maneuver_Type_kMulitiWayLeft = 43,
#   TripDirections_Maneuver_Type_kMulitiWayMiddle = 44,
#   TripDirections_Maneuver_Type_kZuzucheKeepStraight = 45,
#   TripDirections_Maneuver_Type_kZuzucheKeepMainRoadStraight = 46,
#   TripDirections_Maneuver_Type_kZuzucheKeepMainRoadLeft = 47,
#   TripDirections_Maneuver_Type_kZuzucheKeepMainRoadRight = 48
# };

dstDic = {}
if hjson['trip']['status'] == 0:
    # print hjson['data']
    shape = hjson['trip']['legs'][0]['shape']
    # print shape
    # route_shape=data['shape']
    # print route_shape
    shape_str = decode(shape)
    shape_json = shape_str
    maneuvers = hjson['trip']['legs'][0]['maneuvers']

    maneuvers_len = len(maneuvers)
    for i in range(2, maneuvers_len):
        pre_maneuver = maneuvers[i - 1]
        maneuver = maneuvers[i]
        intersections = maneuver['intersections']
        begin_shape_index = maneuver['begin_shape_index']
        end_shape_index = maneuver['end_shape_index']
        pre_begin_shape_index = pre_maneuver['begin_shape_index']
        pre_end_shape_index = pre_maneuver['end_shape_index']
        type = maneuver['type']
        if len(intersections) > 0:
            print("intersection shape:")
            for intersection in intersections:
                print(decode(intersection))
            print("myshape:")
            for index in range(begin_shape_index, end_shape_index + 1):
                print(shape_json[index])
            print("prevshape:")
            for index in range(pre_begin_shape_index, pre_end_shape_index + 1):
                print(shape_json[index])

            print("type:" + str(type))

            print("\r\n")
