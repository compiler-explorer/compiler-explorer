(function () {
    function InstanceFetcher(properties) {
        var self = this;
        var AWS = require('aws-sdk');
        var ec2 = new AWS.EC2({region: properties('region')});
        var tagKey = properties('tagKey');
        var tagValue = properties('tagValue');

        this.onInstances = function (result) {
            var allInstances = [];
            result.Reservations.forEach(function (res) {
                allInstances = allInstances.concat(res.Instances);
            });
            return allInstances.filter(function (reservation) {
                if (reservation.State.Name !== "running") return false;
                return reservation.Tags.some(function (t) {
                    return t.Key == tagKey && t.Value == tagValue;
                });
            });
        };

        this.getInstances = function () {
            return ec2.describeInstances().promise().then(self.onInstances);
        };
    }

    exports.InstanceFetcher = InstanceFetcher;
}).call(this);
