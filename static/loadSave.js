define(function (require) {
    "use strict";
    var $ = require('jquery');

    function LoadSave() {
        this.modal = $('#load-save');
        this.onLoad = _.identity;

        $.getJSON('/source/builtin/list', _.bind(function (list) {
            this.modal.find('.example:visible').remove();
            var examples = this.modal.find('.examples');
            var template = examples.find('.example.template');
            _.each(list, _.bind(function (elem) {
                template
                    .clone()
                    .removeClass('template')
                    .appendTo(examples)
                    .find('a')
                    .text(elem.name)
                    .click(_.bind(function () {
                        this.doLoad(elem.urlpart);
                    }, this));
            }, this));
        }, this));
    }

    LoadSave.prototype.run = function (onLoad) {
        this.onLoad = onLoad;
        this.modal.modal();
    };

    LoadSave.prototype.doLoad = function (urlpart) {
        // TODO: handle errors. consider promises...
        $.getJSON('/source/builtin/load/' + urlpart, _.bind(function (response) {
            this.onLoad(response.file);
        }, this));
        this.modal.modal('hide');
    };

    return {LoadSave: LoadSave};
});