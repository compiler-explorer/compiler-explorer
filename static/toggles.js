define(function (require) {
    "use strict";
    var _ = require('underscore');
    var $ = require('jquery');
    var EventEmitter = require('events');

    function get(domRoot) {
        var result = {};
        _.each(domRoot.find(".btn.active"), function (a) {
            var obj = $(a);
            result[obj.data().bind] = true;
        });
        return result;
    }

    function Toggles(root, state) {
        EventEmitter.call(this);
        this.domRoot = root;
        state = state || get(this.domRoot);
        this.domRoot.find('.btn')
            .click(_.bind(this.onClick, this))
            .each(function () {
                $(this).toggleClass('active', !!state[$(this).data().bind]);
            });
        this.state = get(this.domRoot);
    }

    _.extend(Toggles.prototype, EventEmitter.prototype);

    Toggles.prototype.get = function () {
        return _.clone(this.state);
    };

    Toggles.prototype.onClick = function (event) {
        var button = $(event.currentTarget);
        if (button.hasClass("disabled")) return;
        button.toggleClass('active');
        var before = this.state;
        var after = this.state = get(this.domRoot);
        this.emit('change', before, after);
    };

    return Toggles;
});
